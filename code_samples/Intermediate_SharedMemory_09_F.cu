```cuda
/*********************************************************************
 * Aim of the program (as requested, verbatim):
 * 
 * To fix bank conflicts in a 2D array `shared_mem[ROW][COL]`, padding can be added: 
 * `__shared__ float tile[TILE_SIZE][TILE_SIZE+1];`. Explain how accessing 
 * `tile[threadIdx.y][threadIdx.x]` could cause conflicts, but accessing 
 * `tile[threadIdx.x][threadIdx.y]` is fine.
 * 
 * My thinking:
 * - CUDA shared memory is divided into 32 banks. 
 * - On most GPUs, addresses that differ by multiples of the bank width (e.g. 4 or 8 bytes) 
 *   fall into the same bank. 
 * - When a warp of 32 threads reads or writes from shared memory, if two or more 
 *   threads in the same warp target the same bank, a bank conflict occurs and 
 *   the access is serialized.
 * 
 * In a 2D array defined as `__shared__ float tile[TILE_SIZE][TILE_SIZE+1];`, the
 * second dimension (columns) has `TILE_SIZE+1` entries. This padding ensures
 * that successive rows are offset by an extra element, breaking the pattern
 * that would otherwise map multiple threads in a warp to the same bank.
 * 
 * Access pattern `tile[threadIdx.y][threadIdx.x]`:
 *   - The outer index `threadIdx.y` selects the row, and the inner index
 *     `threadIdx.x` selects the column.
 *   - Threads in the same warp that have the same `threadIdx.y` (i.e. belong to the
 *     same row) will all read/write the same row of shared memory. Because the
 *     stride between consecutive elements in a row is the padded width
 *     `TILE_SIZE+1`, but each thread accesses a different column, the
 *     resulting addresses may still map to the same bank if the padding is
 *     not accounted for correctly. This leads to bank conflicts.
 * 
 * Access pattern `tile[threadIdx.x][threadIdx.y]`:
 *   - Now the outer index `threadIdx.x` selects the row, and the inner index
 *     `threadIdx.y` selects the column.
 *   - Since `threadIdx.x` is unique for each thread in a warp, each thread
 *     accesses a different row. With the padded width, the addresses of these
 *     accesses fall into distinct banks, eliminating conflicts.
 * 
 * This program demonstrates the two access patterns using simple kernels.
 *********************************************************************/

#include <stdio.h>

#define TILE_SIZE 16   // Example tile size

// Kernel that writes to shared memory using tile[threadIdx.y][threadIdx.x]
// This pattern can cause bank conflicts on some GPU architectures
__global__ void write_conflict(float *out) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x = threadIdx.x;
    int y = threadIdx.y;

    // Write a value into shared memory
    tile[y][x] = 1.0f;

    __syncthreads();

    // Read back using the same pattern (potential conflict)
    float val = tile[y][x];

    // Store the result to global memory (flattened index)
    int idx = y * TILE_SIZE + x;
    out[idx] = val;
}

// Kernel that reads from shared memory using tile[threadIdx.x][threadIdx.y]
// This pattern avoids bank conflicts because each thread accesses a distinct row
__global__ void read_no_conflict(float *in, float *out) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x = threadIdx.x;
    int y = threadIdx.y;

    // Load data from global to shared memory (row-major)
    tile[y][x] = in[y * TILE_SIZE + x];

    __syncthreads();

    // Read from shared memory using column-major pattern (no conflict)
    float val = tile[x][y];

    // Store the result to global memory
    int idx = y * TILE_SIZE + x;
    out[idx] = val;
}

int main() {
    int size = TILE_SIZE * TILE_SIZE;
    size_t bytes = size * sizeof(float);

    // Host memory allocation
    float *h_in  = (float*)malloc(bytes);
    float *h_out1 = (float*)malloc(bytes);
    float *h_out2 = (float*)malloc(bytes);

    // Initialize input data
    for (int i = 0; i < size; ++i) h_in[i] = (float)i;

    // Device memory allocation
    float *d_in, *d_out1, *d_out2;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out1, bytes);
    cudaMalloc(&d_out2, bytes);

    // Copy input to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Launch kernels
    dim3 block(TILE_SIZE, TILE_SIZE);
    write_conflict<<<1, block>>>(d_out1);
    read_no_conflict<<<1, block>>>(d_in, d_out2);

    // Copy results back to host
    cudaMemcpy(h_out1, d_out1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out2, d_out2, bytes, cudaMemcpyDeviceToHost);

    // Print a few results for verification
    printf("write_conflict outputs[0..15]: ");
    for (int i = 0; i < 16; ++i) printf("%.1f ", h_out1[i]);
    printf("\n");

    printf("read_no_conflict outputs[0..15]: ");
    for (int i = 0; i < 16; ++i) printf("%.1f ", h_out2[i]);
    printf("\n");

    // Free resources
    cudaFree(d_in);
    cudaFree(d_out1);
    cudaFree(d_out2);
    free(h_in);
    free(h_out1);
    free(h_out2);

    return 0;
}
```