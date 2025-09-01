/*
Aim: Implement matrix-scalar addition.

Thinking:
The goal is to create a small CUDA program that demonstrates adding a scalar value to each element of a matrix.
Key points to cover:
1. Represent the matrix in a contiguous 1D array for simplicity; the element at (row, col) is stored at index `row * cols + col`.
2. Allocate memory on both host (CPU) and device (GPU). Host memory holds the original matrix, device memory holds the matrix during computation.
3. Implement a simple CUDA kernel that iterates over each matrix element in parallel and adds the scalar to that element.
4. Use a grid‑stride loop or calculate the total number of elements and launch enough threads so that every element is processed.
5. Copy the resulting matrix back to host memory and verify the result by printing the matrix before and after the operation.
6. Include basic CUDA error checking to help debug if something goes wrong.
7. The program will work for any matrix size; for demonstration, a small 4x4 matrix is used, but the code is written generically.
8. The program uses standard C/C++ style with `cuda_runtime.h`. All memory is freed before exit.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* CUDA error checking macro */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Kernel to add a scalar to each element of the matrix */
__global__ void matrix_scalar_add(float *d_mat, float scalar, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        d_mat[idx] += scalar;
    }
}

/* Helper function to print a matrix */
void print_matrix(const float *mat, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%6.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

int main(void)
{
    /* Define matrix dimensions */
    const int rows = 4;
    const int cols = 4;
    const int N = rows * cols;

    /* Allocate host memory */
    float *h_mat = (float *)malloc(N * sizeof(float));
    if (!h_mat) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize matrix with some values, e.g., i*cols + j */
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_mat[i * cols + j] = (float)(i * cols + j + 1); // 1, 2, ..., 16
        }
    }

    /* Scalar to add */
    const float scalar = 5.0f;

    printf("Original matrix:\n");
    print_matrix(h_mat, rows, cols);

    /* Allocate device memory */
    float *d_mat = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_mat, N * sizeof(float)));

    /* Copy matrix to device */
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Launch kernel: use 256 threads per block */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_scalar_add<<<blocksPerGrid, threadsPerBlock>>>(d_mat, scalar, rows, cols);
    CUDA_CHECK(cudaGetLastError());   // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_mat, d_mat, N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\nMatrix after adding scalar %f:\n", scalar);
    print_matrix(h_mat, rows, cols);

    /* Clean up */
    CUDA_CHECK(cudaFree(d_mat));
    free(h_mat);

    return EXIT_SUCCESS;
}
