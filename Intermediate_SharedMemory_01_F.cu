```cuda
/*
Aim of the program: Implement naive matrix-vector multiplication, where each thread block computes a portion of the output vector.

Thinking process:
1. Understand the requirement: We need a CUDA program that multiplies a matrix A (M x N) by a vector x (N elements) to produce a result vector y (M elements). Each thread block should compute a contiguous segment of y.
2. Choose kernel design: The simplest approach is one thread per output element. Each thread calculates the dot product of one row of A with the vector x. The thread block size can be 256 (typical) and the grid size computed accordingly.
3. Allocation:
   - Host: Allocate arrays for A, x, and y. Initialize A and x with some values (e.g., random or sequential).
   - Device: Allocate corresponding device arrays and copy data from host to device.
4. Kernel launch:
   - Define block size (e.g., 256 threads per block).
   - Compute grid size as ceil(M / blockSize).
   - Each thread calculates its global row index. If the index is within bounds, it computes the dot product.
5. After kernel execution, copy the result vector y back to host and optionally verify or print some elements.
6. Clean up: free host and device memory.
7. Add error checking after CUDA calls for robustness.
8. Include necessary headers and use `__global__` for kernel definition.
9. Keep code concise but readable with comments explaining each section.
10. Output should be a single .cu file with all code from start to end, no additional text.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Matrix dimensions
#define M 1024   // number of rows
#define N 1024   // number of columns

// Thread block size
#define BLOCK_SIZE 256

// CUDA error checking macro
#define CUDA_CHECK(err)                                    \
    if (err != cudaSuccess) {                              \
        fprintf(stderr, "CUDA error: %s (err_num=%d)\n",    \
                cudaGetErrorString(err), err);              \
        exit(EXIT_FAILURE);                                \
    }

// Naive matrix-vector multiplication kernel
// Each thread computes one element of the output vector y.
__global__ void matVecMulKernel(const float *A, const float *x, float *y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // global row index
    if (row < rows) {
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum += A[row * cols + j] * x[j];
        }
        y[row] = sum;
    }
}

int main(void) {
    // Host memory allocation
    float *h_A = (float *)malloc(M * N * sizeof(float));
    float *h_x = (float *)malloc(N * sizeof(float));
    float *h_y = (float *)malloc(M * sizeof(float));

    if (!h_A || !h_x || !h_y) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    // Initialize input data (for example, sequential values)
    for (int i = 0; i < M * N; ++i) {
        h_A[i] = (float)(i % 100) / 100.0f; // values between 0 and 0.99
    }
    for (int i = 0; i < N; ++i) {
        h_x[i] = (float)(i % 50) / 50.0f; // values between 0 and 0.98
    }

    // Device memory allocation
    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void **)&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_y, M * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch configuration
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    matVecMulKernel<<<gridDim, blockDim>>>(d_A, d_x, d_y, M, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));

    // Optionally, print first few results
    printf("First 10 elements of result vector y:\n");
    for (int i = 0; i < 10 && i < M; ++i) {
        printf("y[%d] = %f\n", i, h_y[i]);
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_A);
    free(h_x);
    free(h_y);

    return 0;
}
```