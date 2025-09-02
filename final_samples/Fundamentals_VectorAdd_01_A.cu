/*
Use `float`s for the vectors instead of `int`s.

Thinking:
I needed a CUDA program that demonstrates element‑wise addition of two vectors, but with the
vectors stored as floating point numbers rather than integers.  
The typical approach is:
  * Allocate host arrays of type `float`.
  * Allocate corresponding device arrays with `cudaMalloc`.
  * Copy the host data to the device.
  * Launch a kernel that computes `C[i] = A[i] + B[i]` for each element.
  * Copy the result back to the host.
  * Verify correctness and clean up resources.

The kernel uses standard 1‑D indexing with blocks and threads.  
Error checking macros are added for robustness.  
The vector size is configurable, but the example uses 1,048,576 elements for demonstration.

This program can be compiled with `nvcc`:
    nvcc -o vector_add vector_add.cu
and run as:
    ./vector_add
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Error checking macro
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Kernel to perform element-wise addition of two float vectors
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;           // Number of elements (1,048,576)
    const int SIZE = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(SIZE);
    float *h_B = (float *)malloc(SIZE);
    float *h_C = (float *)malloc(SIZE);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_B, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_C, SIZE));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());          // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize());     // Wait for kernel to finish

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Verification failed at index %d: %f != %f\n",
                    i, h_C[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition succeeded for %d elements.\n", N);
    } else {
        printf("Vector addition failed.\n");
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
