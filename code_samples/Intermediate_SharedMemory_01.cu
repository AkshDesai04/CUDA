```c
/*
Aim: Write a naive matrix multiplication kernel (C = A * B). Each thread will be responsible for calculating a single element of the output matrix C. This will involve the thread reading one full row from A and one full column from B from global memory.

Thinking:
1. Use a straightforward CUDA kernel where each thread computes one element of the result matrix C.
2. For element (i, j) in C, the thread reads the entire i-th row of A and the entire j-th column of B from global memory.
3. The computation is a simple dot product over the shared dimension K.
4. The kernel will be launched with a grid that covers the dimensions of C; each block will be sized to a small tile (e.g., 16x16 threads).
5. Bounds checking ensures that for non-square or non-multiple-of-block-size matrices, threads that fall outside the matrix do nothing.
6. Host code will allocate and initialize matrices A and B with sample data, transfer them to device memory, launch the kernel, copy the result back, and print a few values for verification.
7. For simplicity, all matrices are square with size N = 512, but the code can be adapted for different sizes by changing N or passing sizes at runtime.
8. Error checking macros are included to catch CUDA runtime errors.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 512              // Size of the square matrices
#define BLOCK_SIZE 16      // Threads per block in each dimension

// CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Naive matrix multiplication kernel
// Each thread computes one element of matrix C
__global__ void matMulKernel(const float *A, const float *B, float *C, int n)
{
    // Calculate the row and column index of the element to compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        // Compute dot product of A's row and B's column
        for (int k = 0; k < n; ++k)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main()
{
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        return EXIT_FAILURE;
    }

    // Initialize input matrices with sample data
    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = 1.0f;          // For simplicity, set all elements to 1.0
        h_B[i] = 1.0f;          // This will make each C element equal to N
    }

    // Allocate device memory
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaCheckError(cudaMalloc((void **)&d_A, bytes));
    cudaCheckError(cudaMalloc((void **)&d_B, bytes));
    cudaCheckError(cudaMalloc((void **)&d_C, bytes));

    // Copy input data from host to device
    cudaCheckError(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Configure grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    matMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaCheckError(cudaGetLastError());   // Check for launch errors

    // Wait for GPU to finish before accessing on host
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("Sample output matrix C (first 5x5 block):\n");
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            printf("%6.1f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```