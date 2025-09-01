/*
Implement in-place subtraction: `A[i] -= B[i]`.

Thinking:
The goal is to perform element‑wise subtraction of two arrays A and B such that each element of A is updated in place:
    A[i] = A[i] - B[i]

To leverage the GPU we will write a CUDA kernel that executes this operation in parallel across the array elements.  
Key design points:

1. **Kernel definition** – A simple element‑wise kernel that checks bounds and updates A[i] directly.
2. **Memory management** – Allocate host arrays, initialize them, copy them to device memory, launch the kernel, copy results back, and free memory.
3. **Error checking** – Wrap CUDA calls with a helper macro to catch and report errors early.
4. **Launch configuration** – Use a reasonable block size (e.g., 256) and compute the required number of blocks to cover the array.
5. **Input size** – Accept the array size as a command‑line argument; fall back to a default size if none provided.
6. **Verification** – After the kernel completes, print the first few elements of the result to confirm correct execution.
7. **Self‑contained** – The file is a complete .cu program that can be compiled with `nvcc` and run directly.

This program is intentionally minimal yet demonstrates the core steps of a CUDA application performing an in‑place subtraction.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel: in-place subtraction A[i] -= B[i]
__global__ void subtractKernel(float *A, const float *B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] -= B[idx];
    }
}

int main(int argc, char *argv[]) {
    // Determine array size
    int N = 1024 * 1024; // Default to 1M elements
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid size provided.\n");
            return EXIT_FAILURE;
        }
    }
    printf("Array size: %d\n", N);

    // Allocate host memory
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    if (!h_A || !h_B) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)(i % 100);       // Example values
        h_B[i] = (float)((i + 1) % 50);  // Example values
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    subtractKernel<<<gridSize, blockSize>>>(d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify (print first 10 results)
    printf("First 10 results after A[i] -= B[i]:\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("A[%d] = %.2f\n", i, h_A[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);

    return 0;
}
