/*
Aim of the program: Generalize the vector addition program to handle vectors of a large size, e.g., 1,048,576 elements. This will require launching multiple blocks. Each thread must calculate its unique global index using blockIdx.x * blockDim.x + threadIdx.x to work on the correct element.

Thinking:
1. We need a CUDA kernel that performs element‑wise addition of two vectors.  
2. The kernel should compute a global index with the formula `blockIdx.x * blockDim.x + threadIdx.x`.  
3. We will set the vector size to 1,048,576 (2^20) and choose a block size that is a common GPU warp size multiple, e.g., 256 threads per block.  
4. The grid size is computed as `(N + blockSize - 1) / blockSize` to cover all elements.  
5. Host code will allocate host vectors, initialize them, allocate device memory, copy data to the device, launch the kernel, copy the result back, verify a few elements, and free resources.  
6. Error checking will be performed after each CUDA API call for robustness.  
7. The program will output a few sample results to confirm correctness.  
8. The code is written in C/C++ with CUDA extensions and compiles as a .cu file.  
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 20)          // 1,048,576 elements
#define BLOCK_SIZE 256       // Threads per block

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void)
{
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));
    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Device memory allocation
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_B, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_C, N * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Determine grid and block dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    vectorAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 results
    printf("First 10 results of vector addition:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    // Reset device
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
