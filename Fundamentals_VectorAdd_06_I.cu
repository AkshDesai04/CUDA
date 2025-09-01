/*
Compare the performance of your SAXPY to the cublasSaxpy function from the cuBLAS library.

Thought Process:
1. We need a CUDA program that implements SAXPY both via a custom kernel and via the cuBLAS library.
2. Allocate host arrays X and Y with random values, allocate device arrays for X and two Y buffers (one for the custom kernel and one for cuBLAS).
3. Implement a simple SAXPY kernel: y[i] = a * x[i] + y[i].
4. Measure execution time of the kernel using cudaEventRecord/cudaEventElapsedTime.
5. Create a cuBLAS handle, call cublasSaxpy on the same X and Y buffer, and measure its time.
6. Verify correctness by comparing results from the custom kernel and cuBLAS; compute the maximum difference.
7. Print both timings and the verification result.
8. Clean up all resources.
9. Include comprehensive error checking macros for CUDA and cuBLAS calls.
10. Ensure the program is self‑contained and can be compiled with nvcc.
*/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

#define CHECK_CUBLAS(call)                                       \
    do {                                                         \
        cublasStatus_t status = call;                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                   \
            fprintf(stderr, "CUBLAS error in %s (%s:%d): %d\n",  \
                    #call, __FILE__, __LINE__, status);          \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

__global__ void saxpyKernel(int N, float a, const float *x, float *y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        y[idx] = a * x[idx] + y[idx];
}

int main()
{
    const int N = 1 << 24;          // 16M elements
    const float alpha = 2.5f;       // scaling factor

    // Host allocations
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));
    float *h_y_custom = (float*)malloc(N * sizeof(float));
    float *h_y_cublas = (float*)malloc(N * sizeof(float));

    // Initialize random data
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(rand()) / RAND_MAX;
        h_y[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device allocations
    float *d_x = nullptr;
    float *d_y_custom = nullptr;
    float *d_y_cublas = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y_custom, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y_cublas, N * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_custom, h_y, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_cublas, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    // Timing variables
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ==================== Custom Kernel ====================
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    CHECK_CUDA(cudaEventRecord(start));
    saxpyKernel<<<blocksPerGrid, threadsPerBlock>>>(N, alpha, d_x, d_y_custom);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_custom = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_custom, start, stop));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_y_custom, d_y_custom, N * sizeof(float), cudaMemcpyDeviceToHost));

    // ==================== cuBLAS SAXPY ====================
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUBLAS(cublasSaxpy(handle, N, &alpha, d_x, 1, d_y_cublas, 1));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_cublas = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_cublas, start, stop));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_y_cublas, d_y_cublas, N * sizeof(float), cudaMemcpyDeviceToHost));

    // ==================== Verification ====================
    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = fabs(h_y_custom[i] - h_y_cublas[i]);
        if (diff > max_diff)
            max_diff = diff;
    }

    // ==================== Output ====================
    printf("Custom Kernel Time: %.3f ms\n", ms_custom);
    printf("cuBLAS Saxpy Time:  %.3f ms\n", ms_cublas);
    printf("Maximum difference between results: %e\n", max_diff);

    // ==================== Cleanup ====================
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y_custom));
    CHECK_CUDA(cudaFree(d_y_cublas));
    free(h_x);
    free(h_y);
    free(h_y_custom);
    free(h_y_cublas);

    return 0;
}
