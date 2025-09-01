/*
Aim: If you launch more threads than elements N, the grid-stride loop will have some threads do no work. Verify this is handled correctly.

Thinking:
- We'll write a CUDA kernel that uses a grid‑stride loop to process an array of N elements.
- We'll intentionally launch a huge number of threads (e.g., 64 blocks * 256 threads/block = 16384 threads) while the array has only 1024 elements.
- Each thread will compute its global index (tid) and then iterate over indices starting from that tid with stride = blockDim.x * gridDim.x.
- The loop condition will ensure that threads whose starting index is beyond N simply skip the loop.
- We'll also have each thread count how many elements it processed, storing that count in a separate array 'cnt' of size totalThreads.
- After kernel execution, we'll copy back the processed array and the counts.
- We will verify that:
  1) All array elements contain the expected value (i+1).
  2) The sum of all counts equals N, confirming that exactly N elements were processed and no extra work was done.
- We'll output a success message if all checks pass, otherwise indicate failure.
- The program uses basic CUDA memory allocation, kernel launch, and synchronization, and includes minimal error checking.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256
#define GRID_SIZE 64  // 64 * 256 = 16384 threads > N

// Simple macro for error checking
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel that processes elements using grid‑stride loop
__global__ void processKernel(int *data, int *cnt, int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    int localCount = 0;
    for (int i = tid; i < n; i += stride) {
        data[i] = i + 1;          // Simple operation: set element to i+1
        localCount++;
    }
    // Store the count for this thread (if we have space)
    if (tid < blockDim.x * gridDim.x) {
        cnt[tid] = localCount;
    }
}

int main(void) {
    int *h_data = (int*)malloc(N * sizeof(int));
    int *h_cnt  = (int*)malloc(GRID_SIZE * BLOCK_SIZE * sizeof(int));

    int *d_data, *d_cnt;
    CUDA_CHECK(cudaMalloc((void**)&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cnt, GRID_SIZE * BLOCK_SIZE * sizeof(int)));

    // Initialize device data to zero
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cnt, 0, GRID_SIZE * BLOCK_SIZE * sizeof(int)));

    // Launch kernel
    processKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_data, d_cnt, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back results
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cnt, d_cnt, GRID_SIZE * BLOCK_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify data
    int data_ok = 1;
    for (int i = 0; i < N; ++i) {
        if (h_data[i] != i + 1) {
            data_ok = 0;
            printf("Data mismatch at index %d: got %d, expected %d\n", i, h_data[i], i + 1);
            break;
        }
    }

    // Verify counts sum to N
    long long total_processed = 0;
    for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; ++i) {
        total_processed += h_cnt[i];
    }
    int count_ok = (total_processed == N);

    // Report
    if (data_ok && count_ok) {
        printf("Success: All %d elements processed correctly with %d threads.\n", N, GRID_SIZE * BLOCK_SIZE);
    } else {
        if (!data_ok) {
            printf("Failure: Data verification failed.\n");
        }
        if (!count_ok) {
            printf("Failure: Count verification failed. Total processed = %lld, expected = %d\n", total_processed, N);
        }
    }

    // Clean up
    free(h_data);
    free(h_cnt);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_cnt));

    return 0;
}
