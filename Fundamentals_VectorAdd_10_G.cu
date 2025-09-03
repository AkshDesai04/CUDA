```cpp
/*

Aim of the program: Calculate and print the speedup factor (CPU time / GPU time).

Thought process:
- The objective is to demonstrate a performance comparison between CPU and GPU execution for a simple, embarrassingly parallel task. 
- Vector addition is chosen because it is trivially parallelizable and has minimal memory overhead, allowing the timing to focus primarily on compute differences rather than data transfer complexities.
- CPU implementation will be a straightforward loop, timed using C++'s high_resolution_clock for nanosecond precision.
- GPU implementation will launch a kernel that performs element-wise addition in parallel. GPU timing will use CUDA events, which are the recommended way to measure elapsed device time with sub-microsecond resolution.
- Memory allocation will be done on both host (using new/delete) and device (using cudaMalloc). Host data will be initialized with a deterministic pattern to avoid randomness affecting timing consistency.
- After both executions, CPU and GPU times will be printed along with the computed speedup factor (CPU time divided by GPU time). This gives a clear quantitative measure of the benefit of GPU acceleration for this task.
- The code will include basic error checking for CUDA API calls to ensure any failures are reported promptly.
- Finally, the program will clean up all allocated resources before exiting.

*/

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__   \
                      << " (" << cudaGetErrorString(err) << ")" << std::endl; \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// Simple CUDA kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const size_t N = 1 << 24; // 16 million elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C_cpu = new float[N];
    float *h_C_gpu = new float[N];

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.5f;
        h_B[i] = static_cast<float>(i) * 1.5f;
    }

    // CPU vector addition
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;

    // Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch GPU kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));

    // Wait for GPU to finish
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Measure GPU time
    float gpu_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results (optional, but good practice)
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        if (abs(h_C_cpu[i] - h_C_gpu[i]) > 1e-5f) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": CPU " << h_C_cpu[i]
                      << " vs GPU " << h_C_gpu[i] << std::endl;
            break;
        }
    }
    if (correct) {
        std::cout << "Result verification: PASSED" << std::endl;
    } else {
        std::cout << "Result verification: FAILED" << std::endl;
    }

    // Compute speedup
    double speedup = cpu_duration.count() / gpu_time_ms;
    std::cout << "Speedup factor (CPU time / GPU time): " << speedup << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```