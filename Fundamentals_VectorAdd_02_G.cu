/*
Implement vector division: C[i] = A[i] / B[i]. Add a check in the kernel to prevent division by zero.
[thinking]
The goal is to create a CUDA program that performs element-wise division of two input arrays A and B,
storing the result in array C. We need to handle the edge case where B[i] == 0 by adding a conditional
check in the kernel: if B[i] is zero, we set C[i] to zero (or another sentinel value). The program
will:
1. Define a kernel `divKernel` that takes device pointers to A, B, and C and the number of elements N.
   Inside, each thread calculates its global index `i` and, if `i < N`, performs the division
   only if B[i] is not zero.
2. In the host code:
   - Allocate host arrays for A, B, and C.
   - Initialize A and B with sample values (including some zeros in B to test the guard).
   - Allocate device memory for A, B, and C.
   - Copy A and B from host to device.
   - Determine block and grid dimensions and launch the kernel.
   - Copy the result C back to host.
   - Print a few elements to verify correctness.
   - Clean up all allocated memory.
3. Include CUDA error checking after each API call and kernel launch.
4. Use a fixed vector size (e.g., N = 1<<20) but print only the first few elements to keep output short.
5. Compile with `nvcc` and run; the program should not crash due to division by zero.

This design keeps the code straightforward and demonstrates proper CUDA programming practices:
kernel launch, memory management, error handling, and a simple division-by-zero guard.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Kernel to perform element-wise division with zero check */
__global__ void divKernel(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float b = B[idx];
        if (b != 0.0f)
            C[idx] = A[idx] / b;
        else
            C[idx] = 0.0f; // or any sentinel value you prefer
    }
}

/* Macro for CUDA error checking */
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main()
{
    const int N = 1 << 20;           // 1,048,576 elements
    const int bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)(i + 1);          // arbitrary values
        h_B[i] = (float)((i % 100) - 50); // includes zeros when i%100==50
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    /* Copy input data to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    divKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for completion

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify and print first few results */
    printf("First 10 results (A / B):\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %.4f (A=%f, B=%f)\n", i, h_C[i], h_A[i], h_B[i]);
    }

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
