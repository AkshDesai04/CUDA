/*
Implement the kernel using a 2D grid of 2D blocks.

Thinking:
The task is to write a CUDA program that demonstrates the use of a 2‑D grid of 2‑D blocks.  
A simple and commonly used example for such a layout is element‑wise addition of two
matrices.  Each thread will process one matrix element; the thread indices are derived
from both the block and grid indices.  The kernel will be launched with a 2‑D grid
(`gridDim.x`, `gridDim.y`) and each block will also be 2‑D (`blockDim.x`, `blockDim.y`).
The host code will allocate device memory, copy sample data from host to device,
invoke the kernel, copy the result back, and perform a small verification.

Key points covered:
* Allocation of host and device arrays.
* Populating input matrices with sample values.
* Using `cudaMalloc`, `cudaMemcpy`.
* Defining a kernel that computes `C[i][j] = A[i][j] + B[i][j]`.
* Choosing block size of (16,16) and computing grid size to cover the matrix.
* Synchronization and error checking.
* Copying results back and printing a few entries.

This code can be compiled with `nvcc`:
    nvcc -o matrix_add matrix_add.cu
and executed to see the result. 
*/

#include <cstdio>
#include <cuda_runtime.h>

#define N 512   // Matrix dimension (N x N)
#define BLOCK_SIZE 16  // Block dimension (BLOCK_SIZE x BLOCK_SIZE)

// Kernel to perform element-wise addition of two matrices
__global__ void matrixAdd(const float* A, const float* B, float* C, int width)
{
    // Compute global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we do not go out of bounds
    if (row < width && col < width)
    {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const int size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N * N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error launching kernel: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify a few results
    printf("Sample output (first 5x5 block):\n");
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            printf("%8.1f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
