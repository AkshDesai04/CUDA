/*
Aim of the program: Modify the program to launch with 32 blocks instead of 8.

My thinking:
- The original problem statement mentioned launching a CUDA kernel with 8 blocks.
- To modify it, we simply change the grid dimension in the kernel launch configuration.
- For demonstration purposes, I will implement a simple vector addition kernel.
- The kernel will be launched with 32 blocks, each containing 256 threads (so a total of 8192 elements).
- The host code will allocate and initialize two input vectors, copy them to the device, launch the kernel, copy the result back, and verify correctness.
- I will use standard CUDA error checking and basic timing to show performance.
- The code is fully self-contained and can be compiled with `nvcc -o vecadd vecadd.cu`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

// Simple error checking macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

int main(void)
{
    const int N = 8192;               // Total number of elements
    const int THREADS_PER_BLOCK = 256; // Threads per block
    const int BLOCKS = 32;             // Launch 32 blocks as requested

    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(BLOCKS);
    vecAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: host=%f, device=%f, expected=%f\n",
                    i, h_C[i], expected, expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition successful. All results match.\n");
    } else {
        printf("Vector addition failed.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
