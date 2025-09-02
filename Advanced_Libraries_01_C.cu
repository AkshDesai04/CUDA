```cuda
/*
Aim of the program:
Read the cuBLAS documentation for cublasSgemm and understand the parameters for transposing matrices and scaling factors (alpha and beta). Use it to compute C = 2.0*A*B + 0.5*C.

Thinking and design:
1. cuBLAS function cublasSgemm performs single-precision general matrix multiplication:
   C = alpha * op(A) * op(B) + beta * C
   where op(A) and op(B) can be either A or A^T depending on the transpose flags.
2. For the target computation C = 2.0*A*B + 0.5*C, we set:
   - alpha = 2.0f
   - beta  = 0.5f
   - transA = CUBLAS_OP_N (no transpose)
   - transB = CUBLAS_OP_N (no transpose)
3. Matrix dimensions:
   Let A be M x K, B be K x N, then C must be M x N.
   For simplicity we use square matrices of size N x N. Thus M = K = N.
4. Workflow:
   a. Allocate host arrays for A, B, C.
   b. Initialize them with some values (e.g., sequential or random).
   c. Allocate device arrays and copy host data to device.
   d. Create a cublas handle.
   e. Call cublasSgemm with appropriate parameters.
   f. Copy result back to host.
   g. Verify by printing a few elements.
   h. Clean up resources.
5. Error handling:
   Use simple macros to check CUDA and cuBLAS status codes.
6. Build instructions:
   g++ -o matmul matmul.cu -lcublas -lcudart
   (or nvcc: nvcc -o matmul matmul.cu -lcublas)
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Macro for cuBLAS error checking
#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t stat = call;                                               \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,   \
                    stat);                                                       \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Function to initialize matrix with simple pattern for verification
void init_matrix(float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = static_cast<float>(i + 1);
}

int main()
{
    const int N = 512; // Matrix size N x N
    const int size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize host matrices
    init_matrix(h_A, N, N);
    init_matrix(h_B, N, N);
    init_matrix(h_C, N, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Scalars alpha and beta
    const float alpha = 2.0f;
    const float beta  = 0.5f;

    // Leading dimensions (since matrices are stored in column-major by cuBLAS)
    const int lda = N;
    const int ldb = N;
    const int ldc = N;

    // Perform C = alpha * A * B + beta * C
    // Since we want standard multiplication (no transpose), use CUBLAS_OP_N
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             d_A, lda,
                             d_B, ldb,
                             &beta,
                             d_C, ldc));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Simple verification: print a few elements of C
    printf("Sample results from C after computation:\n");
    for (int i = 0; i < 5; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```