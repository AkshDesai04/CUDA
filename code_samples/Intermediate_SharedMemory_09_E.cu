```cuda
/*
Write a kernel where `output[i] = shared_mem[threadIdx.x * 32]`. This should cause severe 32-way bank conflicts. Profile it and compare to the stride-1 case.

My approach:

1. I will create two kernels.
   * The first kernel (`conflictKernel`) will load data into shared memory contiguously (`shared_mem[tid] = input[i];`) and then each thread will read from `shared_mem[tid * 32]`.  
     Because the shared memory is addressed as 32‑bank memory, all indices `tid * 32` map to bank 0 (`(4 * tid * 32) / 4 % 32 == 0`), causing a full 32‑way conflict.
   * The second kernel (`strideKernel`) will load data contiguously and read back from the same contiguous location (`shared_mem[tid]`). This has no bank conflict.

2. Shared memory sizes:
   * For the conflict kernel, the maximum index accessed is `(BLOCK_SIZE-1)*32`.  
     Therefore I allocate `BLOCK_SIZE * 32` ints.
   * For the stride kernel, only the first `BLOCK_SIZE` entries are used; I allocate `BLOCK_SIZE` ints.

3. I will time both kernels using CUDA events and print the time in milliseconds and the throughput (elements per second). The outputs of the two kernels will be compared for correctness.

4. The program will be self‑contained, allocate host input data, copy it to device, launch both kernels, copy back the results, and finally clean up.

5. I added a `CUDA_CHECK` macro for error handling and used `cudaDeviceSynchronize()` to ensure kernels finish before timing and copying results.

6. I used `BLOCK_SIZE = 256`, `N = 1<<20` (1M elements) as a typical workload.

7. The code is written in a single .cu file and can be compiled with `nvcc -o conflict_test conflict_test.cu`.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define BLOCK_SIZE 256
#define N (1 << 20)   // 1M elements

__global__ void conflictKernel(int *output, const int *input)
{
    extern __shared__ int s[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < N) {
        s[tid] = input[idx];          // load contiguously
    }
    __syncthreads();
    if (idx < N) {
        // Read from the same bank (bank 0), causing 32‑way conflict
        output[idx] = s[tid * 32];
    }
}

__global__ void strideKernel(int *output, const int *input)
{
    extern __shared__ int s[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < N) {
        s[tid] = input[idx];          // load contiguously
    }
    __syncthreads();
    if (idx < N) {
        // Read from contiguous address, no conflict
        output[idx] = s[tid];
    }
}

int main()
{
    // Allocate host memory
    int *h_input = (int*)malloc(N * sizeof(int));
    int *h_output_conflict = (int*)malloc(N * sizeof(int));
    int *h_output_stride = (int*)malloc(N * sizeof(int));

    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_input[i] = rand() % 1000;
    }

    // Allocate device memory
    int *d_input, *d_output_conflict, *d_output_stride;
    CUDA_CHECK(cudaMalloc((void**)&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_output_conflict, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_output_stride, N * sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    // Define execution configuration
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + threads.x - 1) / threads.x);

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch conflict kernel
    CUDA_CHECK(cudaEventRecord(start));
    conflictKernel<<<blocks, threads, BLOCK_SIZE * 32 * sizeof(int)>>>(
        d_output_conflict, d_input);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_conflict;
    CUDA_CHECK(cudaEventElapsedTime(&ms_conflict, start, stop));

    // Launch stride kernel
    CUDA_CHECK(cudaEventRecord(start));
    strideKernel<<<blocks, threads, BLOCK_SIZE * sizeof(int)>>>(
        d_output_stride, d_input);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_stride;
    CUDA_CHECK(cudaEventElapsedTime(&ms_stride, start, stop));

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_output_conflict, d_output_conflict,
                          N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_stride, d_output_stride,
                          N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify correctness
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (h_output_conflict[i] != h_output_stride[i]) {
            mismatches++;
            if (mismatches <= 10) {
                printf("Mismatch at %d: conflict=%d stride=%d\n",
                       i, h_output_conflict[i], h_output_stride[i]);
            }
        }
    }
    if (mismatches == 0) {
        printf("Results match.\n");
    } else {
        printf("Total mismatches: %d\n", mismatches);
    }

    // Print timing results
    printf("Conflict kernel time:   %.3f ms\n", ms_conflict);
    printf("Stride kernel time:     %.3f ms\n", ms_stride);
    printf("Speedup (stride / conflict): %.2f\n", ms_conflict / ms_stride);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_conflict));
    CUDA_CHECK(cudaFree(d_output_stride));
    free(h_input);
    free(h_output_conflict);
    free(h_output_stride);

    return 0;
}
```