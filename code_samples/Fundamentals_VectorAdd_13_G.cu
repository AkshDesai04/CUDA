/*
Implement `C[i] = (A[i] == B[i]) ? 1 : 0`.

Thinking:
- The aim is to create a CUDA program that performs an element‑wise comparison between two integer arrays A and B.
- For each index i, if A[i] equals B[i], set C[i] to 1; otherwise set C[i] to 0.
- The program will allocate memory on both host and device, copy data to the device, launch a kernel that does the comparison, and copy the result back.
- We'll use a simple kernel where each thread handles one element.
- The program includes basic error handling and prints a few results for verification.
- Size of the arrays can be changed; for this example, we use 1024 elements.
- Thread block size of 256 is chosen for good occupancy.
- The code is fully self‑contained and can be compiled with nvcc: nvcc -o compare compare.cu
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for element-wise comparison
__global__ void compare_arrays(const int *A, const int *B, int *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = (A[idx] == B[idx]) ? 1 : 0;
    }
}

// Utility macro for checking CUDA errors
#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(void)
{
    const int n = 1024;          // Number of elements
    size_t bytes = n * sizeof(int);

    // Allocate host memory
    int *h_A = (int *)malloc(bytes);
    int *h_B = (int *)malloc(bytes);
    int *h_C = (int *)malloc(bytes);

    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with random data
    for (int i = 0; i < n; ++i)
    {
        h_A[i] = rand() % 10;
        h_B[i] = rand() % 10;
    }

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    compare_arrays<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CHECK_CUDA(cudaGetLastError());   // Check for launch errors
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Print first 10 results
    printf("Index\tA[i]\tB[i]\tC[i]\n");
    for (int i = 0; i < 10 && i < n; ++i)
    {
        printf("%d\t%d\t%d\t%d\n", i, h_A[i], h_B[i], h_C[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
