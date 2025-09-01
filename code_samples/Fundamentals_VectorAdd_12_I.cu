/*
Aim: Write a kernel to set a matrix to the identity matrix (1s on the diagonal, 0s elsewhere).

Thinking:
- The goal is to provide a simple CUDA program that demonstrates how to write a kernel that initializes a matrix to an identity matrix.
- The matrix is stored in row-major order, and the kernel will be launched with a 2D grid of blocks and a 2D block of threads so that each thread handles one matrix element.
- Each thread computes its global row and column indices using blockIdx, blockDim, threadIdx. 
- Boundary checks are performed to ensure that the indices are within the matrix dimensions; this makes the kernel robust for non-square matrices and for grid/ block sizes that do not divide the matrix exactly.
- For an element at (row, col):
    * If row == col, set the element to 1.0f (float) to represent the identity.
    * Otherwise, set the element to 0.0f.
- The host code allocates memory on both the host and the device, launches the kernel, copies the result back, and prints the matrix to verify correctness.
- CUDA error checking is implemented via a helper macro to catch and report any errors after API calls.
- The matrix size is chosen to be 4x4 for demonstration purposes, but the code can handle arbitrary dimensions by changing the constants or passing them as arguments.
- The program is self‑contained, uses only the CUDA runtime API, and can be compiled with `nvcc identity.cu -o identity`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Kernel to set a matrix to identity */
__global__ void set_identity(float *matrix, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        matrix[idx] = (row == col) ? 1.0f : 0.0f;
    }
}

/* Utility function to print a matrix */
void print_matrix(const float *matrix, int rows, int cols)
{
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            printf("%6.1f ", matrix[r * cols + c]);
        }
        printf("\n");
    }
}

int main(void)
{
    /* Matrix dimensions */
    const int rows = 4;
    const int cols = 4;
    const size_t size = rows * cols * sizeof(float);

    /* Allocate host memory */
    float *h_matrix = (float *)malloc(size);
    if (h_matrix == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Allocate device memory */
    float *d_matrix;
    CUDA_CHECK(cudaMalloc((void **)&d_matrix, size));

    /* Configure kernel launch parameters */
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    /* Launch kernel */
    set_identity<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, rows, cols);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost));

    /* Print the resulting matrix */
    printf("Identity matrix (%dx%d):\n", rows, cols);
    print_matrix(h_matrix, rows, cols);

    /* Clean up */
    CUDA_CHECK(cudaFree(d_matrix));
    free(h_matrix);

    return 0;
}
