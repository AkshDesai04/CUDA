```c
/*
Use cuBLAS to perform a matrix-vector multiplication (`cublasSgemv`).

The goal of this program is to demonstrate how to perform a simple matrix-vector
multiplication on the GPU using the cuBLAS library.  We will:

1. Create a small matrix `A` of size M×N and a vector `x` of size N on the host.
2. Allocate corresponding device memory for `A`, `x`, and the result vector `y`.
3. Copy the data from host to device.
4. Create a cuBLAS handle and call `cublasSgemv` to compute `y = alpha * A * x + beta * y`.
5. Copy the result back to the host and print it.
6. Clean up all allocated resources.

This example uses single-precision floating point (`float`) and the basic
`cublasSgemv` routine, which is a wrapper around the BLAS `sgemv` function.
The program includes simple error checking macros for both CUDA runtime
calls and cuBLAS API calls to ensure that any failure is reported.

The matrix and vector are initialized with deterministic values so that the
output can be verified manually.  The `alpha` and `beta` scalars are set to 1.0f
and 0.0f respectively, which gives the standard matrix-vector product `y = A * x`.

Compile with:
    nvcc -o sgemv_example sgemv_example.cu -lcublas
Run with:
    ./sgemv_example
*/

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Macro for checking CUDA runtime errors
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Macro for checking cuBLAS API errors
#define CHECK_CUBLAS(call)                                         \
    do {                                                           \
        cublasStatus_t status = call;                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                    \
            fprintf(stderr, "cuBLAS error at %s:%d\n",             \
                    __FILE__, __LINE__);                          \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

int main(void) {
    // Define matrix dimensions
    const int M = 4;   // number of rows
    const int N = 5;   // number of columns

    // Scalars for cublasSgemv
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Allocate host memory for matrix A (column-major order) and vectors x, y
    float h_A[M * N];
    float h_x[N];
    float h_y[M];  // will store the result

    // Initialize matrix A and vector x with deterministic values
    for (int col = 0; col < N; ++col) {
        for (int row = 0; row < M; ++row) {
            h_A[col * M + row] = (float)(row + col * M + 1); // e.g., 1,2,...
        }
    }
    for (int i = 0; i < N; ++i) {
        h_x[i] = (float)(i + 1);  // 1.0, 2.0, ...
    }
    // Initialize y to zeros
    for (int i = 0; i < M; ++i) {
        h_y[i] = 0.0f;
    }

    // Allocate device memory
    float *d_A, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(float) * M * N));
    CHECK_CUDA(cudaMalloc((void**)&d_x, sizeof(float) * N));
    CHECK_CUDA(cudaMalloc((void**)&d_y, sizeof(float) * M));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, sizeof(float) * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, sizeof(float) * M, cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Perform y = alpha * A * x + beta * y
    // Note: cuBLAS expects column-major storage, which matches our layout
    // Parameters:
    //   trans: no transpose
    //   m: number of rows of A (M)
    //   n: number of columns of A (N)
    //   alpha, beta
    //   A: device pointer
    //   lda: leading dimension of A (here M)
    //   x: device pointer
    //   incx: stride for x (1)
    //   y: device pointer
    //   incy: stride for y (1)
    CHECK_CUBLAS(cublasSgemv(handle,
                             CUBLAS_OP_N,   // No transpose
                             M,             // m
                             N,             // n
                             &alpha,        // alpha
                             d_A,           // A
                             M,             // lda
                             d_x,           // x
                             1,             // incx
                             &beta,         // beta
                             d_y,           // y
                             1));           // incy

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_y, d_y, sizeof(float) * M, cudaMemcpyDeviceToHost));

    // Print the result vector
    printf("Result vector y = A * x:\n");
    for (int i = 0; i < M; ++i) {
        printf("y[%d] = %f\n", i, h_y[i]);
    }

    // Clean up
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return 0;
}
```