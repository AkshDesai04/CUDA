/*
Aim: Read the cuBLAS documentation for `cublasSgemm` and understand the parameters for transposing matrices and scaling factors (alpha and beta). Use it to compute `C = 2.0*A*B + 0.5*C`.

Thinking:
- `cublasSgemm` performs a single-precision general matrix-matrix multiplication with optional transposition and scaling.
- The function signature (simplified) is:
      cublasStatus_t cublasSgemm(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 int m, int n, int k,
                                 const float *alpha,
                                 const float *A, int lda,
                                 const float *B, int ldb,
                                 const float *beta,
                                 float *C, int ldc);
  where `m` is the number of rows of the output matrix C,
  `n` is the number of columns of C,
  `k` is the inner dimension.
- The matrix data is stored in column-major order (as expected by cuBLAS). The `lda`, `ldb`, and `ldc` parameters are the leading dimensions (i.e. the stride between consecutive columns).
- Transposition flags: `CUBLAS_OP_N` means no transpose, `CUBLAS_OP_T` means transpose.
- The scalars `alpha` and `beta` are applied as `C = alpha * op(A) * op(B) + beta * C`.

To compute `C = 2.0*A*B + 0.5*C` we set:
  - `transa = CUBLAS_OP_N`
  - `transb = CUBLAS_OP_N`
  - `alpha = 2.0f`
  - `beta  = 0.5f`
- We will create small matrices (2x2) for demonstration.
- The program will:
  1. Allocate host matrices and initialize them.
  2. Allocate device memory and copy the matrices over.
  3. Create a cuBLAS handle.
  4. Call `cublasSgemm`.
  5. Copy the result back to the host and print it.
  6. Clean up resources.

This example will compile with `nvcc` and requires linking against the cuBLAS library (`-lcublas`).
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Macro for checking CUDA errors
#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                     \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Macro for checking cuBLAS errors
#define CHECK_CUBLAS(call)                                                        \
    do {                                                                          \
        cublasStatus_t stat = call;                                               \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error in %s (%s:%d)\n",                      \
                    #call, __FILE__, __LINE__);                                  \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Helper function to print a matrix stored in column-major order
void printMatrix(const char* name, const float* mat, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // In column-major, element (i,j) is at index i + j*rows
            printf("%8.3f ", mat[i + j * rows]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void) {
    // Define matrix dimensions
    const int M = 2;   // rows of A and C
    const int N = 2;   // columns of B and C
    const int K = 2;   // columns of A, rows of B

    // Allocate host matrices (column-major)
    float h_A[M * K] = {1.0f, 2.0f,
                        3.0f, 4.0f}; // 2x2
    float h_B[K * N] = {5.0f, 6.0f,
                        7.0f, 8.0f}; // 2x2
    float h_C[M * N] = {9.0f, 10.0f,
                        11.0f, 12.0f}; // 2x2

    printf("Input matrices:\n");
    printMatrix("A", h_A, M, K);
    printMatrix("B", h_B, K, N);
    printMatrix("C (initial)", h_C, M, N);

    // Device pointers
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Scalars
    const float alpha = 2.0f;
    const float beta  = 0.5f;

    // Leading dimensions
    int lda = M; // since matrices are stored column-major and no transpose
    int ldb = K;
    int ldc = M;

    // Perform C = alpha * A * B + beta * C
    // Since we use no transpose: CUBLAS_OP_N for both A and B
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             M,      // number of rows of C
                             N,      // number of columns of C
                             K,      // shared dimension
                             &alpha,
                             d_A, lda,
                             d_B, ldb,
                             &beta,
                             d_C, ldc));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Result matrix C after computation:\n");
    printMatrix("C (final)", h_C, M, N);

    // Clean up
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
