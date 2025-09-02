```cuda
/*
 * Aim of the program: Implement matrix subtraction and multiplication
 *
 * Thought process and design decisions:
 * 1. The program is written in CUDA C++ (.cu) and showcases two basic linear algebra
 *    operations: element‑wise matrix subtraction and matrix multiplication.
 * 2. Two CUDA kernels are provided:
 *    - `matSubtractKernel`: Each thread computes a single element of the result
 *      matrix `C = A - B`. The kernel uses a simple 2‑D grid and block
 *      configuration derived from the matrix dimensions.
 *    - `matMulKernel`: Implements the classic naïve matrix multiplication
 *      algorithm (`C = A * B`). Each thread computes one element of the
 *      product matrix by iterating over the shared dimension `k`. For simplicity
 *      and clarity, no shared‑memory tiling or optimization is used.
 * 3. The host code:
 *    - Allocates example matrices `A` and `B` on the host, initializes them
 *      with deterministic values (row‑major incremental numbers), and prints
 *      them for debugging.
 *    - Allocates corresponding device memory for `A`, `B`, and result matrices
 *      `C_sub` (subtraction result) and `C_mul` (multiplication result).
 *    - Copies `A` and `B` to the device.
 *    - Launches the subtraction kernel followed by the multiplication kernel.
 *    - Copies the results back to host memory.
 *    - Prints the resulting matrices.
 * 4. Error checking: All CUDA API calls are wrapped with a helper function
 *    `checkCudaError` that prints an informative message on failure and aborts.
 * 5. Matrix size: The example uses a fixed 4x4 matrix for demonstration. The
 *    code can be easily extended to larger sizes by changing the `N` constant.
 * 6. Memory layout: Matrices are stored in row‑major order, so indexing is
 *    `index = row * width + col`. This matches the typical C/C++ layout and
 *    simplifies both host and device code.
 * 7. Compilation: The code can be compiled with `nvcc -o mat_ops mat_ops.cu`.
 *    It will produce an executable that prints the input matrices and the
 *    results of subtraction and multiplication.
 * 8. Limitations: No advanced optimizations (shared memory, coalesced
 *    accesses, loop unrolling) are applied. The focus is on clarity and
 *    correctness rather than performance.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

// Kernel for element‑wise matrix subtraction: C = A - B
__global__ void matSubtractKernel(const float *A, const float *B, float *C,
                                  int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] - B[idx];
    }
}

// Kernel for matrix multiplication: C = A * B
__global__ void matMulKernel(const float *A, const float *B, float *C,
                             int A_rows, int A_cols, int B_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // col of C

    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_cols; ++k) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

// Utility function to print a matrix
void printMatrix(const float *mat, int rows, int cols, const char *name)
{
    printf("%s =\n", name);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main()
{
    const int N = 4;  // Size of square matrices
    const int rows = N;
    const int cols = N;

    size_t bytes = rows * cols * sizeof(float);

    // Allocate host matrices
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_sub = (float *)malloc(bytes);
    float *h_C_mul = (float *)malloc(bytes);

    // Initialize input matrices with deterministic values
    for (int i = 0; i < rows * cols; ++i) {
        h_A[i] = static_cast<float>(i);              // 0, 1, 2, ...
        h_B[i] = static_cast<float>(i * 2);          // 0, 2, 4, ...
    }

    printMatrix(h_A, rows, cols, "A");
    printMatrix(h_B, rows, cols, "B");

    // Allocate device matrices
    float *d_A, *d_B, *d_C_sub, *d_C_mul;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C_sub, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C_mul, bytes));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define block and grid dimensions for subtraction
    dim3 blockSizeSub(16, 16);
    dim3 gridSizeSub((cols + blockSizeSub.x - 1) / blockSizeSub.x,
                     (rows + blockSizeSub.y - 1) / blockSizeSub.y);

    // Launch subtraction kernel
    matSubtractKernel<<<gridSizeSub, blockSizeSub>>>(d_A, d_B, d_C_sub, rows, cols);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Define block and grid dimensions for multiplication
    dim3 blockSizeMul(16, 16);
    dim3 gridSizeMul((cols + blockSizeMul.x - 1) / blockSizeMul.x,
                     (rows + blockSizeMul.y - 1) / blockSizeMul.y);

    // Launch multiplication kernel
    matMulKernel<<<gridSizeMul, blockSizeMul>>>(d_A, d_B, d_C_mul,
                                               rows, cols, cols);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_C_sub, d_C_sub, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_mul, d_C_mul, bytes, cudaMemcpyDeviceToHost));

    printMatrix(h_C_sub, rows, cols, "C = A - B");
    printMatrix(h_C_mul, rows, cols, "C = A * B");

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C_sub);
    free(h_C_mul);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C_sub));
    CHECK_CUDA(cudaFree(d_C_mul));

    return 0;
}
```