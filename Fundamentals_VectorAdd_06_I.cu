/*
Aim: Compare the performance of your SAXPY to the `cublasSaxpy` function from the cuBLAS library.

Thinking:
1. The program must implement a simple SAXPY operation (y = alpha * x + y) in two ways:
   - A custom CUDA kernel that performs the operation on the GPU.
   - The cuBLAS library function `cublasSaxpy`, which is a highly optimized routine for the same operation.
2. To compare performance, we need to:
   - Allocate host and device memory for large vectors (e.g., N = 1e7).
   - Initialize vectors x and y with random data and set a scalar alpha.
   - Copy data to the device.
   - Run the custom kernel and time it with CUDA events.
   - Run `cublasSaxpy` and time it with CUDA events.
   - Copy the results back to the host and compare them to ensure correctness.
   - Report the elapsed times and compute the speedup or performance difference.
3. The code will use single‑precision floats because cuBLAS’s `cublasSaxpy` operates on floats (`cublasSaxpy` for double would be `cublasDaxpy`).
4. Error checking will be done via a helper macro that prints an error message and exits on failure.
5. The program will be compiled with `nvcc` and linked against the cuBLAS library (e.g., `nvcc -lcublas saxpy_comparison.cu -o saxpy_comparison`).
6. The entire program, including the comment with the aim and reasoning, will be contained in a single `.cu` file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>

#define CUDA_CHECK(err) do { \
    cudaError_t err__ = (err); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(err) do { \
    cublasStatus_t err__ = (err); \
    if (err__ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, err__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Simple SAXPY kernel: y = alpha * x + y
__global__ void saxpy_kernel(int N, float alpha, const float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] += alpha * x[idx];
    }
}

int main(void) {
    const int N = 1 << 24; // ~16 million elements
    const float alpha = 2.5f;

    // Allocate host memory
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));
    float *h_y_custom = (float*)malloc(N * sizeof(float));
    float *h_y_cublas = (float*)malloc(N * sizeof(float));

    if (!h_x || !h_y || !h_y_custom || !h_y_cublas) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // random in [-1,1]
        h_y[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    // Copy initial y to both host buffers for comparison later
    memcpy(h_y_custom, h_y, N * sizeof(float));
    memcpy(h_y_cublas, h_y, N * sizeof(float));

    // Allocate device memory
    float *d_x = NULL;
    float *d_y = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // -----------------------------
    // Run custom SAXPY kernel
    // -----------------------------
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaEventRecord(start, 0));
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, alpha, d_x, d_y);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds_custom = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_custom, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_y_custom, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // -----------------------------
    // Run cuBLAS SAXPY
    // -----------------------------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Need to reset d_y to original h_y values
    CUDA_CHECK(cudaMemcpy(d_y, h_y_cublas, N * sizeof(float), cudaMemcpyHostToDevice));

    // Measure cublas time
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUBLAS_CHECK(cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds_cublas = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_cublas, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_y_cublas, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // -----------------------------
    // Verify correctness
    // -----------------------------
    double max_error = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = fabs(h_y_custom[i] - h_y_cublas[i]);
        if (err > max_error) max_error = err;
    }
    printf("Maximum error between custom kernel and cuBLAS: %e\n", max_error);

    // -----------------------------
    // Report timings
    // -----------------------------
    printf("Custom kernel time:   %f ms\n", milliseconds_custom);
    printf("cuBLAS SAXPY time:    %f ms\n", milliseconds_cublas);
    if (milliseconds_cublas > 0.0) {
        printf("Speedup (cuBLAS over custom): %.2f\n", milliseconds_custom / milliseconds_cublas);
    }

    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x);
    free(h_y);
    free(h_y_custom);
    free(h_y_cublas);

    return EXIT_SUCCESS;
}
