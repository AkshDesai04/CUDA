```c
/*
Use cuBLAS to perform a vector dot product (cublasSdot).
This comment block explains the intent and the design of the program.
The goal is to demonstrate how to use the cuBLAS library to compute
the dot product of two single‑precision vectors on the GPU.

Design decisions:
1. Host and device memory allocation:
   - Allocate two vectors a and b of length N on the host.
   - Allocate device memory d_a and d_b for the vectors.
2. Initialization:
   - Fill host vectors with simple data (e.g., a[i] = i+1, b[i] = 2*i+1).
3. cuBLAS usage:
   - Create a cuBLAS handle.
   - Copy host vectors to device memory.
   - Call cublasSdot to compute the dot product on the GPU.
   - Copy the result back to host and print it.
4. Error handling:
   - Use helper macros to check CUDA and cuBLAS API return values.
   - Ensure resources are freed even on failure.
5. Clean up:
   - Destroy the cuBLAS handle.
   - Free device and host memory.
6. Compilation:
   - Compile with nvcc, linking against cublas: nvcc -lcublas vector_dot.cu -o vector_dot
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 1000

// Helper macro to check CUDA API calls
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",       \
                    __FILE__, __LINE__, static_cast<int>(err),                 \
                    cudaGetErrorName(err), cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Helper macro to check cuBLAS API calls
#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error at %s:%d code=%d\n",                 \
                    __FILE__, __LINE__, static_cast<int>(status));             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void) {
    float *h_a = NULL, *h_b = NULL;
    float *d_a = NULL, *d_b = NULL;
    float result = 0.0f;

    // Allocate host memory
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    if (!h_a || !h_b) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i + 1);          // 1, 2, 3, ...
        h_b[i] = static_cast<float>(2 * i + 1);      // 1, 3, 5, ...
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, N * sizeof(float)));

    // Copy host vectors to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Perform dot product: result = a · b
    // Note: cublasSdot uses the signature:
    // cublasStatus_t cublasSdot(cublasHandle_t handle, int n, const float *x, int incx,
    //                           const float *y, int incy, float *result)
    CHECK_CUBLAS(cublasSdot(handle, N, d_a, 1, d_b, 1, &result));

    // Destroy cuBLAS handle
    CHECK_CUBLAS(cublasDestroy(handle));

    // Free device memory
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    // Print the result
    printf("Dot product of vectors a and b: %f\n", result);

    // Free host memory
    free(h_a);
    free(h_b);

    // Reset device (optional)
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```