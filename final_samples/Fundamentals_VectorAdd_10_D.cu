/*
For the CPU timing, use `std::chrono::high_resolution_clock`.

[Thinking:
We will implement a simple CUDA vector addition program that demonstrates CPU timing using std::chrono::high_resolution_clock and GPU timing using CUDA events. The program will:
1. Allocate and initialize host vectors A and B with random values.
2. Allocate device memory and copy the data from host to device.
3. Launch a vector addition kernel to compute C = A + B on the GPU.
4. Copy the result back to host memory.
5. Verify the result.
6. Measure CPU-side timings (initialization, memory allocation, host-to-device and device-to-host copies) using std::chrono::high_resolution_clock.
7. Measure GPU-side kernel execution time using cudaEventRecord and cudaEventElapsedTime.
The program will print the timing results and a success/failure message. All CUDA error checking is handled with a simple macro. The code is fully self-contained and can be compiled with nvcc as a .cu file. */

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>

// Simple CUDA error checking macro
#define CUDA_CHECK(err)                                            \
    do {                                                           \
        cudaError_t err_ = (err);                                  \
        if (err_ != cudaSuccess) {                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":"       \
                      << __LINE__ << " - "                         \
                      << cudaGetErrorString(err_) << std::endl;    \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// CUDA kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const int bytes = N * sizeof(float);

    // CPU timing: initialization
    auto cpu_start = std::chrono::high_resolution_clock::now();

    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.5f;
        h_B[i] = static_cast<float>(i) * 2.0f;
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> init_time = cpu_end - cpu_start;

    // Allocate device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // CPU timing: host-to-device copy
    cpu_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> h2d_time = cpu_end - cpu_start;

    // GPU timing: kernel launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    // CPU timing: device-to-host copy
    cpu_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> d2h_time = cpu_end - cpu_start;

    // Verify results
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": "
                      << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            break;
        }
    }

    // Print timing results
    std::cout << "CPU initialization time: " << init_time.count() << " ms" << std::endl;
    std::cout << "CPU host-to-device copy time: " << h2d_time.count() << " ms" << std::endl;
    std::cout << "GPU kernel execution time: " << gpu_ms << " ms" << std::endl;
    std::cout << "CPU device-to-host copy time: " << d2h_time.count() << " ms" << std::endl;

    std::cout << "Result verification: " << (correct ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
