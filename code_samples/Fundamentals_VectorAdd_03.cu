```cpp
/*
 * Aim of the program: Modify your large vector addition program to handle input sizes that are not an exact multiple of the block size (e.g., 1,000,000 elements). Add a boundary check inside the kernel (if (index < N) { ... }) to prevent threads from accessing memory out of bounds.
 *
 * Thinking:
 * - The original vector addition program likely launched a kernel with grid dimensions derived from the vector size divided by block size, assuming that size is a multiple of block size.
 * - When the size is not a multiple of the block size, the last block will contain threads that refer to indices beyond the valid range, causing out‑of‑bounds memory access.
 * - The fix is to compute the grid size as ceil(N / blockSize), so that there are enough threads to cover all elements, and then guard each thread with a boundary check: if (index < N) { ... }.
 * - The rest of the program remains largely unchanged: allocate host and device memory, initialize host arrays, copy to device, launch kernel, copy result back, and optionally verify correctness.
 * - We'll use a simple block size of 256 and compute the grid dimension as (N + blockSize - 1) / blockSize.
 * - The kernel will perform the addition only for valid indices.
 * - We'll include a simple error checking macro for CUDA calls.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void vecAddKernel(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    const int N = 1000000; // Example size not a multiple of typical block size
    const int bytes = N * sizeof(float);
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernel
    vecAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify result
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
            correct = false;
            printf("Mismatch at index %d: %f + %f != %f\n", i, h_a[i], h_b[i], h_c[i]);
            break;
        }
    }
    printf("Vector addition %s.\n", correct ? "PASSED" : "FAILED");

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```