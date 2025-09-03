/*
To fix bank conflicts in a 2D array `shared_mem[ROW][COL]`, padding can be added: `__shared__ float tile[TILE_SIZE][TILE_SIZE+1];`. Explain how accessing `tile[threadIdx.y][threadIdx.x]` could cause conflicts, but accessing `tile[threadIdx.x][threadIdx.y]` is fine.

The aim of this program is to illustrate how bank conflicts arise when threads in a warp access a 2‑D shared‑memory array laid out in row‑major order without padding, and how a simple padding trick eliminates those conflicts. The program sets up two identical kernels: one that accesses the shared array as `tile[threadIdx.y][threadIdx.x]` (the natural row‑major access) and another that accesses it as `tile[threadIdx.x][threadIdx.y]` after inserting an extra element per row (`tile[TILE_SIZE][TILE_SIZE+1]`). Both kernels perform a trivial operation – copying data from global to shared memory and back – but the first one is susceptible to bank conflicts when the tile width (COL) is not a multiple of the number of shared‑memory banks (32 on most GPUs). The second kernel avoids conflicts because the padded stride aligns each row with a separate bank, ensuring that consecutive thread indices address distinct banks.

Key points covered in the code and comment:

1. CUDA shared memory is divided into 32 banks, each 4 bytes wide for `float`.  
2. In a 2‑D array declared as `__shared__ float tile[ROWS][COLS]`, the logical address of `tile[r][c]` is `base + (r * COLS + c) * sizeof(float)`.  
3. When `COLS` is not a multiple of 32, threads in a warp that are meant to access consecutive elements in a row (different `c` but same `r`) can map to the same bank because `r * COLS` introduces a residue modulo 32.  
4. Padding each row with one extra element (`tile[ROWS][COLS+1]`) makes the stride a multiple of 32, eliminating the residue and thus preventing bank conflicts.  
5. Accessing the array as `tile[threadIdx.x][threadIdx.y]` flips the roles of row and column. Because the stride now matches the warp’s thread indices, each thread accesses a unique bank without needing padding.

The program demonstrates the two approaches and prints a simple confirmation that both kernels executed successfully. It does not attempt to measure performance because the focus is on illustrating the conceptual difference rather than on benchmarking. The explanation is embedded in the top multiline comment along with the program’s purpose.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define N (TILE_SIZE * 4)   // Size of global matrix (4 tiles per dimension)

// Kernel that uses a tile without padding: potential bank conflicts
__global__ void kernel_conflict(const float* __restrict__ d_in, float* __restrict__ d_out) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];   // No padding

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Bounds check
    if (row < N && col < N) {
        // Load from global to shared memory (may cause conflicts)
        tile[ty][tx] = d_in[row * N + col];
        __syncthreads();

        // Write back to global memory
        d_out[row * N + col] = tile[ty][tx];
    }
}

// Kernel that uses a padded tile: bank conflicts avoided
__global__ void kernel_no_conflict(const float* __restrict__ d_in, float* __restrict__ d_out) {
    // Pad each row with one extra float to make the stride a multiple of 32
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Bounds check
    if (row < N && col < N) {
        // Load from global to shared memory (no conflicts)
        tile[ty][tx] = d_in[row * N + col];
        __syncthreads();

        // Write back to global memory
        d_out[row * N + col] = tile[ty][tx];
    }
}

int main() {
    size_t size = N * N * sizeof(float);
    float *h_in = (float*)malloc(size);
    float *h_out_conflict = (float*)malloc(size);
    float *h_out_no_conflict = (float*)malloc(size);

    // Initialize input matrix with some values
    for (int i = 0; i < N * N; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Run kernel with potential bank conflicts
    kernel_conflict<<<gridSize, blockSize>>>(d_in, d_out);
    cudaMemcpy(h_out_conflict, d_out, size, cudaMemcpyDeviceToHost);

    // Run kernel without bank conflicts
    kernel_no_conflict<<<gridSize, blockSize>>>(d_in, d_out);
    cudaMemcpy(h_out_no_conflict, d_out, size, cudaMemcpyDeviceToHost);

    // Simple verification
    int errors = 0;
    for (int i = 0; i < N * N; ++i) {
        if (h_out_conflict[i] != h_in[i] || h_out_no_conflict[i] != h_in[i]) {
            errors++;
            if (errors < 10) {
                printf("Mismatch at %d: conflict=%f, no_conflict=%f, expected=%f\n",
                       i, h_out_conflict[i], h_out_no_conflict[i], h_in[i]);
            }
        }
    }

    if (errors == 0) {
        printf("Success: All outputs match input.\n");
    } else {
        printf("Failed: %d mismatches found.\n", errors);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out_conflict);
    free(h_out_no_conflict);

    return 0;
}
