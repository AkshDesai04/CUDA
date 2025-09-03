/*
 * Aim of the program:
 * Include the HtoD and DtoH data transfer times in the GPU timing. Now which is faster for a moderately sized vector? This shows the impact of transfer overhead.
 *
 * Thinking:
 * - Create a moderately sized vector (10 million floats) on the host.
 * - Perform a simple element-wise operation (multiplication by 2) both on the CPU (sequential loop) and on the GPU (CUDA kernel).
 * - Measure CPU time using std::chrono high_resolution_clock.
 * - For GPU:
 *   * Measure HtoD transfer time using cudaEvent_t.
 *   * Measure kernel execution time using cudaEvent_t.
 *   * Measure DtoH transfer time using cudaEvent_t.
 *   * Sum these three to get total GPU time including transfer overhead.
 * - Print all timing components.
 * - Compare CPU time vs GPU total time and output which is faster.
 * - Include error checking for CUDA calls.
 * - Keep the code self-contained and compilable as a single .cu file.
 */

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(err)                                                        \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(EXIT_FAILURE);                                                   \
    }

// Simple CUDA kernel to multiply each element by 2
__global__ void vectorMultiply(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

int main() {
    const int N = 10 * 1000 * 1000; // 10 million elements (moderately sized)
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_in = new float[N];
    float* h_out_cpu = new float[N];
    float* h_out_gpu = new float[N];

    // Initialize input vector with some values
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i) * 0.001f;
    }

    // CPU computation timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        h_out_cpu[i] = h_in[i] * 2.0f;
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU time: " << cpu_duration.count() << " ms\n";

    // GPU memory allocation
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, size));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size));

    // Create CUDA events for timing
    cudaEvent_t h2d_start, h2d_end, kernel_start, kernel_end, d2h_start, d2h_end;
    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_end));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_end));
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_end));

    // HtoD transfer timing
    CUDA_CHECK(cudaEventRecord(h2d_start));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(h2d_end));
    CUDA_CHECK(cudaEventSynchronize(h2d_end));

    float h2d_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_time, h2d_start, h2d_end));
    std::cout << "Host-to-Device transfer time: " << h2d_time << " ms\n";

    // Kernel execution timing
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_CHECK(cudaEventRecord(kernel_start));
    vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(kernel_end));
    CUDA_CHECK(cudaEventSynchronize(kernel_end));

    float kernel_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end));
    std::cout << "Kernel execution time: " << kernel_time << " ms\n";

    // DtoH transfer timing
    CUDA_CHECK(cudaEventRecord(d2h_start));
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2h_end));
    CUDA_CHECK(cudaEventSynchronize(d2h_end));

    float d2h_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_time, d2h_start, d2h_end));
    std::cout << "Device-to-Host transfer time: " << d2h_time << " ms\n";

    // Total GPU time including transfers
    float total_gpu_time = h2d_time + kernel_time + d2h_time;
    std::cout << "Total GPU time (including transfers): " << total_gpu_time << " ms\n";

    // Compare CPU and GPU times
    if (total_gpu_time < cpu_duration.count()) {
        std::cout << "GPU is faster for this vector size.\n";
    } else {
        std::cout << "CPU is faster for this vector size.\n";
    }

    // Clean up
    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_end));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_end));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_end));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    delete[] h_in;
    delete[] h_out_cpu;
    delete[] h_out_gpu;

    return 0;
}
