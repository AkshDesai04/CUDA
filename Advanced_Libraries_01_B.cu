/*
Aim: Use cuBLAS to perform a vector dot product (cublasSdot).

Thinking:
The goal is to demonstrate how to use the cuBLAS library to compute the dot product of two single-precision floating point vectors.  cuBLAS provides the routine `cublasSdot`, which takes a handle, vector length, pointers to device vectors, strides, and a device pointer to store the result.  The steps required are:
1. Allocate host memory for two example vectors.
2. Allocate device memory for the vectors and copy the data from host to device.
3. Create a cuBLAS handle and call `cublasSdot` with the appropriate parameters.
4. Copy the resulting scalar back to the host.
5. Clean up all resources (device memory, cuBLAS handle, and host memory).
6. Print the result to verify correctness.

Edge cases to consider:
- Checking return status from all CUDA and cuBLAS API calls.
- Properly freeing resources in case of errors.
- Using proper stride arguments (incx, incy) which are usually 1 for contiguous arrays.

The example uses small vectors of length 5 with known values so that the expected dot product can be manually verified.  This code is intended to be compiled with NVCC and run on a system with a CUDA-capable GPU and the cuBLAS library installed.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Macro to check CUDA errors
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error [%s:%d] : %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// Macro to check cuBLAS errors
#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error [%s:%d] : %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

int main(void) {
    const int n = 5; // vector length
    float h_x[n] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float h_y[n] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float h_result = 0.0f;

    float *d_x = NULL;
    float *d_y = NULL;
    float *d_result = NULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_result, sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Perform dot product: result = d_x ⋅ d_y
    CUBLAS_CHECK(cublasSdot(handle, n, d_x, 1, d_y, 1, d_result));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    // Print the result
    printf("Dot product of vectors: %f\n", h_result);

    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}
