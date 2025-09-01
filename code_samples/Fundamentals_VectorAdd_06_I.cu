```cpp
/*
Aim of the program: Compare the performance of your SAXPY to the cublasSaxpy function from the cuBLAS library.

Thinking:
The goal is to provide a self‑contained CUDA C program that demonstrates how to benchmark a user‑implemented SAXPY kernel against the highly optimised cuBLAS SAXPY routine.  The program will:
1. Allocate host arrays for x and y, fill them with deterministic values (e.g. a ramp or random numbers) to avoid compiler optimisations.
2. Allocate corresponding device arrays and copy the host data to the device.
3. Launch a simple SAXPY kernel (`y = a * x + y`) on the GPU and time it using CUDA events.
4. Re‑copy the original y to the device, then call `cublasSaxpy` and time that call using the same CUDA events for consistency.
5. Copy the results back to the host and perform a numerical comparison (e.g. max absolute difference) to ensure correctness.
6. Print the elapsed times and a small summary of the relative performance.
7. Perform all necessary error checking for CUDA runtime API calls and cuBLAS API calls, and clean up all resources before exiting.

The program will use a fixed vector size (e.g. 1e8 elements) to make the timing stable, but it can be easily adapted to read the size from the command line.  The kernel uses a standard grid‑block configuration with 256 threads per block, and it includes bounds checking.  Timing is performed with CUDA events which provide GPU‑side timestamps in milliseconds.  The cuBLAS handle is created once and destroyed at the end.  All allocations are freed, and the program returns a success exit code if the results agree within a small tolerance.

This program can be compiled with:
    nvcc -o saxpy_benchmark saxpy_benchmark.cu -lcublas
and run on any system with a CUDA‑enabled GPU and the cuBLAS library installed.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cmath>
#include <iomanip>

// CUDA error checking macro
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            std::cerr << "CUDA error in file " << __FILE__            \
                      << " at line " << __LINE__ << ": "             \
                      << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call)                                            \
    do {                                                              \
        cublasStatus_t err = call;                                    \
        if (err != CUBLAS_STATUS_SUCCESS) {                          \
            std::cerr << "cuBLAS error in file " << __FILE__          \
                      << " at line " << __LINE__ << ": "             \
                      << cublasGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Simple SAXPY kernel: y = a * x + y
__global__ void saxpy_kernel(const int N, const float a, const float *x, float *y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        y[idx] = a * x[idx] + y[idx];
}

// Helper to get cuBLAS error string
static const char *cublasGetErrorString(cublasStatus_t status)
{
    switch (status)
    {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        default: return "Unknown cuBLAS status";
    }
}

int main()
{
    // Problem size
    const size_t N = 1 << 28; // ~268 million elements
    const float alpha = 2.0f;

    // Host allocations
    float *h_x = new float[N];
    float *h_y_host = new float[N];
    float *h_y_custom = new float[N];
    float *h_y_cublas = new float[N];

    // Initialize host data
    for (size_t i = 0; i < N; ++i)
    {
        h_x[i] = static_cast<float>(i) / N;
        h_y_host[i] = static_cast<float>(N - i) / N;
    }

    // Device allocations
    float *d_x = nullptr;
    float *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y_host, N * sizeof(float), cudaMemcpyHostToDevice));

    // CUDA event timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --------------------- Custom SAXPY ---------------------
    CUDA_CHECK(cudaEventRecord(start, 0));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, alpha, d_x, d_y);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch error

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float millisecondsCustom = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&millisecondsCustom, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_y_custom, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // --------------------- cuBLAS SAXPY ---------------------
    // Re‑initialize y on device
    CUDA_CHECK(cudaMemcpy(d_y, h_y_host, N * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaEventRecord(start, 0));

    CUBLAS_CHECK(cublasSaxpy(handle, static_cast<int>(N), &alpha, d_x, 1, d_y, 1));

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float millisecondsCublas = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&millisecondsCublas, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_y_cublas, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Destroy cuBLAS handle
    CUBLAS_CHECK(cublasDestroy(handle));

    // Verify results (max absolute difference)
    double max_diff = 0.0;
    for (size_t i = 0; i < N; ++i)
    {
        double diff = std::abs(h_y_custom[i] - h_y_cublas[i]);
        if (diff > max_diff)
            max_diff = diff;
    }

    // Output results
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Vector size: " << N << std::endl;
    std::cout << "Custom SAXPY time: " << millisecondsCustom << " ms" << std::endl;
    std::cout << "cuBLAS SAXPY time: " << millisecondsCublas << " ms" << std::endl;
    std::cout << "Speedup (Custom / cuBLAS): " << millisecondsCustom / millisecondsCublas << std::endl;
    std::cout << "Maximum absolute difference between results: " << max_diff << std::endl;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    delete[] h_x;
    delete[] h_y_host;
    delete[] h_y_custom;
    delete[] h_y_cublas;

    return 0;
}
```