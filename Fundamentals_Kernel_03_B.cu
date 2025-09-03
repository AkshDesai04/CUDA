```cuda
/*
Aim of the program: Modify the program to launch with 128 threads instead of 64.

Thinking:
The original version of this CUDA example used 64 threads per block when launching the
kernel that performs a simple element‑wise vector addition. To satisfy the new
requirement, the block size must be changed to 128 threads. The grid size is then
recalculated to cover the entire input array. Everything else – memory allocation,
initialisation, kernel logic, result verification, and cleanup – remains unchanged.

Key changes:
1. `dim3 blockSize(128);` instead of `dim3 blockSize(64);`.
2. The calculation of `gridSize` uses the updated block size.
3. No other modifications are required because the kernel is written in a way
   that works for any block size as long as `gridSize` covers all elements.

This program demonstrates the change clearly and verifies correctness by
printing a few sample output values.
*/

#include <stdio.h>
#include <cuda.h>

#define N (1024)  // Size of the vectors

// Simple kernel performing element-wise addition: c[i] = a[i] + b[i]
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    __func__, __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void) {
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));

    // Initialise host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Device pointers
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_c, N * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch parameters
    const int threadsPerBlock = 128;                     // Modified from 64 to 128
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results (print first 5 elements)
    printf("First 5 results of vector addition:\n");
    for (int i = 0; i < 5; ++i) {
        printf("c[%d] = %f\n", i, h_c[i]);
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```