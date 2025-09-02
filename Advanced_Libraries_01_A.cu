/*
Use cuBLAS to perform a matrix-vector multiplication (cublasSgemv).

Thinking:
- We need a self-contained .cu file that demonstrates cublasSgemv usage.
- The program will:
    1. Create a small example matrix A (m x n) and vector x (n).
    2. Allocate device memory and copy data to GPU.
    3. Create cublas handle.
    4. Call cublasSgemv with appropriate parameters.
    5. Copy result vector y back to host.
    6. Print the result.
- Include error checking for CUDA and cuBLAS calls.
- Use single precision floats.
- The matrix A will be stored in column-major order (as expected by cuBLAS).
- Use alpha=1.0f, beta=0.0f.
- Ensure proper cleanup of resources.

Implementation details:
- Use #define CUDA_CHECK etc.
- Example values: m=4, n=3.
- Provide comments for clarity.
*/

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) { cudaCheck((err), __FILE__, __LINE__); }
inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s:%d)\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CUBLAS_CHECK(err) { cublasCheck((err), __FILE__, __LINE__); }
inline void cublasCheck(cublasStatus_t err, const char* file, int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        const char* errStr;
        switch (err) {
            case CUBLAS_STATUS_NOT_INITIALIZED: errStr = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
            case CUBLAS_STATUS_ALLOC_FAILED:    errStr = "CUBLAS_STATUS_ALLOC_FAILED"; break;
            case CUBLAS_STATUS_INVALID_VALUE:   errStr = "CUBLAS_STATUS_INVALID_VALUE"; break;
            case CUBLAS_STATUS_ARCH_MISMATCH:   errStr = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
            case CUBLAS_STATUS_MAPPING_ERROR:   errStr = "CUBLAS_STATUS_MAPPING_ERROR"; break;
            case CUBLAS_STATUS_EXECUTION_FAILED:errStr = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
            case CUBLAS_STATUS_INTERNAL_ERROR:  errStr = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
            default:                            errStr = "CUBLAS_STATUS_UNKNOWN_ERROR"; break;
        }
        fprintf(stderr, "cuBLAS Error: %s (%s:%d)\n", errStr, file, line);
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Matrix dimensions
    const int m = 4; // rows
    const int n = 3; // columns

    // Host memory allocation
    float h_A[m * n]; // column-major
    float h_x[n];
    float h_y[m];

    // Initialize matrix A and vector x with example values
    // For simplicity, fill with sequential values
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            h_A[col * m + row] = static_cast<float>(row + col * m + 1); // some pattern
        }
    }

    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i + 1);
    }

    // Device memory allocation
    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_A, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, m * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Parameters for cublasSgemv
    cublasOperation_t trans = CUBLAS_OP_N; // no transpose
    float alpha = 1.0f;
    float beta = 0.0f;

    // Launch cublasSgemv
    // Note: cuBLAS expects column-major order, which matches our storage
    CUBLAS_CHECK(cublasSgemv(handle,
                             trans,
                             m,
                             n,
                             &alpha,
                             d_A,
                             m,            // leading dimension of A
                             d_x,
                             1,            // stride of x
                             &beta,
                             d_y,
                             1));          // stride of y

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, m * sizeof(float), cudaMemcpyDeviceToHost));

    // Print result
    printf("Result vector y (size %d):\n", m);
    for (int i = 0; i < m; ++i) {
        printf("y[%d] = %f\n", i, h_y[i]);
    }

    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return 0;
}
