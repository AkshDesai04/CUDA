/*
Aim: Explain in a comment why GPUs excel at this type of 'embarrassingly parallel' problem.

Thinking:
1. The prompt requests an explanation inside a comment, but also instructs us to provide a CUDA (.cu) file that implements a program.
2. The simplest embarrassingly parallel task is element‑wise vector addition: each element can be processed independently with no data dependency on other elements.
3. GPUs have many cores (threads) and are designed for data‑parallel workloads where the same operation is applied to large data sets.
4. Because there is no communication or synchronization required between threads in this problem, the GPU can launch a large number of threads, each doing a simple addition, and the hardware’s SIMT architecture will execute them efficiently.
5. The memory hierarchy (global, shared, L1/L2 caches) and coalesced memory accesses further accelerate such tasks.
6. We’ll write a minimal CUDA program that adds two arrays on the GPU and then copies the result back to the host, demonstrating the embarrassingly parallel nature and why GPUs excel at it.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// CUDA kernel for element‑wise vector addition
__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;              // 1M elements
    const int size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy inputs from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel with enough blocks to cover all elements
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify a few results
    for (int i = 0; i < 5; ++i)
    {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
