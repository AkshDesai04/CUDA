```cuda
/*
Handle large vectors with boundary checks.
Thinking:
We want to create a CUDA program that demonstrates handling large vectors, i.e., vectors that may exceed the total number of threads in a single block. We will use a kernel that adds two vectors element-wise into a third vector. The kernel will perform boundary checks to ensure that each thread only accesses valid indices. We will allocate large vectors on the host, copy them to device, launch the kernel with a 1‑D grid of blocks, then copy back and optionally verify a few elements. We will also include error checking macros to capture CUDA API errors. The code will be a single .cu file that can be compiled with nvcc. The program will print some basic information and may include simple timing using cudaEvent.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Kernel for element-wise addition of two large vectors
__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N) {
    // Calculate global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Boundary check: ensure we don't access beyond the array
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    // Size of the vectors (choose a large size, e.g., 100 million elements)
    const size_t N = 100 * 1000 * 1000; // 100 million
    const size_t bytes = N * sizeof(float);

    printf("Vector size: %zu elements (%zu bytes per array)\n", N, bytes);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Determine kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);

    printf("Launching kernel with %d blocks of %d threads each.\n", blocksPerGrid, threadsPerBlock);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: check first 5 elements
    int correct = 1;
    for (int i = 0; i < 5; ++i) {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            printf("Verification failed at index %d: got %f, expected %f\n", i, h_C[i], expected);
            correct = 0;
            break;
        }
    }
    if (correct) {
        printf("Verification passed for first 5 elements.\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("Program completed successfully.\n");
    return 0;
}
```