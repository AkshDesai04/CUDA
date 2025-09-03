/*
Implement matrix-scalar addition.
  
The goal of this program is to demonstrate how to perform a matrix-scalar
addition using CUDA.  The algorithm is straightforward:
  1. Allocate a matrix on the host and fill it with some values.
  2. Allocate a matrix on the device and copy the host matrix to the device.
  3. Launch a kernel where each thread adds the scalar value to one element
     of the matrix.
  4. Copy the result back to the host and display it.
  
Key points considered while writing the code:
  * Use a 1‑D kernel launch where each thread handles a single matrix element.
    The global index `idx` is computed from block and thread indices.
  * Perform bounds checking inside the kernel to avoid out‑of‑bounds
    accesses for the last thread block.
  * Provide a small helper macro `CUDA_CHECK` for error handling.
  * Allocate device memory with `cudaMalloc` and copy data with
    `cudaMemcpy`.
  * After the kernel finishes, copy the data back to the host and free
    both host and device memory.
  * For demonstration purposes, the matrix is printed before and after
    the addition when its size is small.
  
The program compiles to a `.cu` file and can be built with `nvcc`:
  nvcc -o matrix_scalar matrix_scalar.cu
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Helper macro to check CUDA runtime API calls
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",       \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),           \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel to add a scalar value to each element of a matrix
__global__ void add_scalar(float *d_matrix, int total_elements, float scalar)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        d_matrix[idx] += scalar;
    }
}

// Function to print a matrix stored in row-major order
void print_matrix(const char *title, const float *matrix, int rows, int cols)
{
    printf("%s:\n", title);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            printf("%6.2f ", matrix[r * cols + c]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    // Matrix dimensions
    const int rows = 4;   // For demonstration; can be changed
    const int cols = 5;
    const int total = rows * cols;

    // Allocate host matrix
    float *h_matrix = (float *)malloc(total * sizeof(float));
    if (!h_matrix) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Seed random number generator and fill host matrix with values
    srand((unsigned)time(NULL));
    for (int i = 0; i < total; ++i) {
        h_matrix[i] = (float)(rand() % 10); // values between 0 and 9
    }

    // Scalar to add
    const float scalar = 3.5f;

    // Print original matrix
    print_matrix("Original matrix", h_matrix, rows, cols);

    // Allocate device memory
    float *d_matrix = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_matrix, total * sizeof(float)));

    // Copy matrix from host to device
    CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix, total * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int blockSize = 256;
    const int gridSize = (total + blockSize - 1) / blockSize;

    // Launch kernel
    add_scalar<<<gridSize, blockSize>>>(d_matrix, total, scalar);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_matrix, d_matrix, total * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Print result matrix
    print_matrix("Result matrix (after adding scalar)", h_matrix, rows, cols);

    // Clean up
    CUDA_CHECK(cudaFree(d_matrix));
    free(h_matrix);

    return 0;
}
