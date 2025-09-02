/*
 * Aim of the program:
 * Explain why reading `s_tile[threadIdx.y][threadIdx.x]` is fine but writing `s_tile[threadIdx.y][threadIdx.x]` could cause bank conflicts if `threadIdx.y` is the faster-changing index.
 *
 * Thinking:
 * Shared memory in CUDA is divided into 32 banks.  A bank conflict occurs when multiple threads in a warp access the same bank.
 * The bank number for a 32‑bit word is calculated as `(address / 4) % 32`.  If the memory layout causes all threads in a warp to map
 * to the same bank, the accesses are serialized and performance drops.
 *
 * In a 2‑D shared array declared as `__shared__ float s_tile[BLOCK_SIZE][BLOCK_SIZE];`, the memory layout is row major:
 * address(row,col) = base + ((row * BLOCK_SIZE) + col) * sizeof(float).
 * If `threadIdx.y` is the faster‑changing index (i.e. each thread in a warp increments `y` while `x` stays constant until wrap),
 * then for a fixed `threadIdx.x` the accessed addresses differ by `BLOCK_SIZE` words.
 * When `BLOCK_SIZE` is a multiple of 32 (typical warp size), the offset between consecutive threads is a multiple of 32 words,
 * so `(offset / 4) % 32` is the same for all threads.  All threads in the warp hit the same bank.
 *
 * For **reads** this can sometimes be tolerated because the hardware can broadcast a single bank’s value to all threads
 * that read from the same address; the cost is less severe than a write conflict.
 * For **writes**, however, all threads in a warp must acquire exclusive access to the bank.  If they all write to the same bank,
 * the hardware serializes the writes, causing a significant slowdown.
 *
 * This program demonstrates the pattern.  It loads data into shared memory using `s_tile[threadIdx.y][threadIdx.x]`,
 * then performs a simple reduction that writes back to global memory.  The comments highlight the potential bank conflict
 * scenario for writes when `threadIdx.y` is the fastest dimension.  Compile with `-arch=sm_70` (or higher) to observe
 * the behavior on a real device.
 *
 * Note: This code is purely illustrative; it does not perform any timing or conflict detection.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32  // Must be a multiple of 32 to illustrate the bank conflict

__global__ void bank_conflict_demo(const float *A, float *B, int N)
{
    // Allocate shared memory tile
    __shared__ float s_tile[BLOCK_SIZE][BLOCK_SIZE];

    // Compute global index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N)
    {
        // Load from global memory into shared memory
        // This read is fine because each thread accesses a distinct word.
        s_tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }

    __syncthreads();

    // Simple operation: add 1.0f to each element
    // Write back to global memory
    // If threadIdx.y is the faster-changing index, the write may cause a bank conflict
    if (x < N && y < N)
    {
        B[y * N + x] = s_tile[threadIdx.y][threadIdx.x] + 1.0f;
    }
}

int main(void)
{
    const int N = 1024;
    const int size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);

    // Initialize input data
    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = (float)i;
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    bank_conflict_demo<<<gridDim, blockDim>>>(d_A, d_B, N);

    // Copy result back to host
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // Verify a few elements
    for (int i = 0; i < 5; ++i)
    {
        int idx = i * N + i;
        printf("h_B[%d] = %f\n", idx, h_B[idx]);
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}
