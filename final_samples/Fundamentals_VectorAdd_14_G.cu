/*
Aim of the program: Implement vector addition for `long long int`.

Thinking:
To implement vector addition for 64‑bit integers using CUDA, I will:
1. Define the size of the vectors (e.g., N = 1<<20) and allocate host memory for two input vectors (A and B) and one output vector (C).
2. Fill A and B with deterministic values so that the result can be verified easily.
3. Allocate corresponding device memory (dA, dB, dC) using cudaMalloc.
4. Copy the host arrays A and B to the device with cudaMemcpy.
5. Write a kernel `vectorAdd` that takes pointers to the device arrays and performs element‑wise addition:
   - Each thread calculates its global index `idx = blockIdx.x * blockDim.x + threadIdx.x`.
   - If `idx < n`, it computes `C[idx] = A[idx] + B[idx]`.
6. Launch the kernel with a suitable grid and block configuration (e.g., 256 threads per block).
7. After kernel execution, copy the result array C back to host memory.
8. Verify correctness by printing the first few results.
9. Free all allocated device and host memory.
10. Include basic error checking for CUDA calls for robustness.

The kernel uses `long long` (64‑bit signed integer) which is supported on CUDA devices, and the host code uses the same type. The program is self‑contained and can be compiled with `nvcc`.*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel for element‑wise addition of two long long int arrays
__global__ void vectorAdd(const long long *A, const long long *B, long long *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// Utility macro for checking CUDA API calls
#define CUDA_CHECK(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                             \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),          \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

int main(void)
{
    const int N = 1 << 20;           // Size of vectors (1,048,576 elements)
    const int bytes = N * sizeof(long long);

    // Allocate host memory
    long long *h_A = (long long*)malloc(bytes);
    long long *h_B = (long long*)malloc(bytes);
    long long *h_C = (long long*)malloc(bytes);

    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = i;                 // Example: A[i] = i
        h_B[i] = 2 * i;             // Example: B[i] = 2*i
    }

    // Allocate device memory
    long long *d_A = NULL;
    long long *d_B = NULL;
    long long *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy host input to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());   // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("Sample results:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("C[%d] = %lld (expected %lld)\n", i, h_C[i], h_A[i] + h_B[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    printf("Vector addition completed successfully.\n");
    return EXIT_SUCCESS;
}
