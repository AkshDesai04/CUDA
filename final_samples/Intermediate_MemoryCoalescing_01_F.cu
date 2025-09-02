/*
Implement the kernel using a 2D grid of 2D blocks.

Thinking:

- The requirement is to use a 2D grid of 2D blocks in the kernel launch configuration.
- A simple demonstration kernel is matrix addition: C = A + B.
- Each thread will handle one element of the output matrix.
- We set block dimensions to (16,16) so each block is 16x16 threads.
- The grid dimensions are calculated to cover the full matrix size (M rows, N columns).
- CUDA memory allocation, copying, and deallocation are performed on the host.
- Basic error checking is included to ensure successful CUDA calls.
- The program outputs the result matrix to the console for verification.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

__global__ void matrixAddKernel(const float *A, const float *B, float *C, int rows, int cols)
{
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Y-dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x; // X-dimension

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

void printMatrix(const float *mat, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%6.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

int main()
{
    const int rows = 8;   // Number of rows in matrices
    const int cols = 8;   // Number of columns in matrices
    const int size = rows * cols * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize input matrices
    for (int i = 0; i < rows * cols; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x,
                 (rows + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    matrixAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Print result
    printf("Matrix A:\n");
    printMatrix(h_A, rows, cols);
    printf("\nMatrix B:\n");
    printMatrix(h_B, rows, cols);
    printf("\nMatrix C = A + B:\n");
    printMatrix(h_C, rows, cols);

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
