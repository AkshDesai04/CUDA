```cpp
/*
Write a multi-threaded CPU version of vector addition (e.g., using OpenMP) and compare it to the GPU version.

Thinking:

- The goal is to implement vector addition on both CPU and GPU, measure their execution times, and compare the results.
- For the CPU implementation we will use OpenMP to parallelize the loop. 
- For the GPU implementation we will write a simple CUDA kernel that adds corresponding elements.
- We need to allocate host memory for the input vectors and for the CPU and GPU output vectors.
- We will initialize the input vectors with deterministic values (sin and cos) so we can validate correctness.
- Timing:
    * CPU: std::chrono::high_resolution_clock
    * GPU: cudaEvent_t for accurate timing on device
- After running both kernels we will copy the GPU result back to the host and compare it against the CPU result to check for numerical differences.
- We will output the execution times and the maximum absolute error.
- The program will accept an optional command line argument for vector size; otherwise a default large size will be used to see a measurable speedup.
- We include error checking macros for CUDA calls.
- The code will compile with nvcc and requires linking against OpenMP (`-Xcompiler -fopenmp`).
*/

#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call)                                           \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// CUDA kernel for vector addition
__global__ void vecAddKernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// CPU implementation using OpenMP
void vecAddCPU(const float* A, const float* B, float* C, int N) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// Helper to initialize vectors
void initVectors(float* A, float* B, int N) {
    for (int i = 0; i < N; ++i) {
        A[i] = std::sin(static_cast<float>(i));
        B[i] = std::cos(static_cast<float>(i));
    }
}

// Helper to compute maximum absolute difference
float maxAbsDiff(const float* X, const float* Y, int N) {
    float maxDiff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = std::fabs(X[i] - Y[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    return maxDiff;
}

int main(int argc, char* argv[]) {
    int N = 1 << 24; // Default 16 million elements
    if (argc > 1) {
        N = std::atoi(argv[1]);
        if (N <= 0) {
            std::cerr << "Invalid vector size.\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "Vector size: " << N << "\n";

    // Allocate host memory
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C_cpu = new float[N];
    float* h_C_gpu = new float[N];

    // Initialize input vectors
    initVectors(h_A, h_B, N);

    // CPU vector addition
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vecAddCPU(h_A, h_B, h_C_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU time (OpenMP): " << cpu_time_ms << " ms\n";

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // GPU vector addition timing with cudaEvent
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaEventRecord(start));
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpu_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));

    std::cout << "GPU time (CUDA): " << gpu_time_ms << " ms\n";

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    float max_diff = maxAbsDiff(h_C_cpu, h_C_gpu, N);
    std::cout << "Maximum absolute difference between CPU and GPU results: " << max_diff << "\n";

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;

    return 0;
}
```