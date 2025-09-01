/*
Aim: Modify the program to launch with 128 threads instead of 64.

Thinking:
- The original program used a kernel launch configuration of <<<numBlocks, 64>>>.
- To increase parallelism, we change the number of threads per block to 128.
- The rest of the program remains unchanged: a simple vector addition kernel that
  adds two arrays element-wise.  
- We ensure that the grid dimension is computed correctly to cover all N elements.
- We include basic CUDA error checking and memory allocation for a self‑contained example.
- The program compiles as a .cu file and can be built with `nvcc`:
      nvcc -o vector_add vector_add.cu
- Running the program will perform the addition on the GPU and print a few sample results.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000000          // Number of elements in the vectors
#define THREADS_PER_BLOCK 128  // Updated from 64 to 128

// CUDA kernel for element-wise addition
__global__ void vecAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(err)                                                      \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",          \
                __FILE__, __LINE__, err, cudaGetErrorName(err),              \
                cudaGetErrorString(err));                                    \
        exit(EXIT_FAILURE);                                                  \
    }

int main()
{
    // Host allocations
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c = (float *)malloc(N * sizeof(float));

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Device allocations
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_c, N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threadsPerBlock = THREADS_PER_BLOCK;          // 128 threads per block
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few results
    for (int i = 0; i < 5; ++i) {
        printf("c[%d] = %f\n", i, h_c[i]);
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
