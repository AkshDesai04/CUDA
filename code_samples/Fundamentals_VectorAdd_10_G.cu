/*
Calculate and print the speedup factor (CPU time / GPU time).

The aim of the program is to demonstrate how to measure and compare the execution time of a simple computational task
performed on the CPU versus the GPU using CUDA.  The chosen task is an element‑wise addition of two large float vectors,
which is a common benchmark for illustrating GPU acceleration.

The program proceeds as follows:

1.  Define a problem size (number of elements) – here we use 1 << 20 (1 million elements) to provide a workload large enough
    to reveal a noticeable difference between CPU and GPU execution times.

2.  Allocate host memory for two input vectors (A and B) and one output vector (C).  The input vectors are filled with random
    float values.

3.  Measure the CPU time:
    - Start a high‑resolution clock (std::chrono::high_resolution_clock).
    - Perform the element‑wise addition on the CPU in a simple for loop.
    - Stop the clock and compute the elapsed time in milliseconds.

4.  Allocate device memory for the vectors, copy the input data from host to device, and create CUDA events for timing the GPU
    kernel execution.

5.  Measure the GPU time:
    - Record a start event.
    - Launch a CUDA kernel that performs the vector addition in parallel.
    - Record an end event.
    - Synchronize on the stop event and compute the elapsed time between the two events in milliseconds.

6.  Copy the result back to host memory (not timed as part of the kernel execution).

7.  Compute the speedup factor as (CPU time / GPU time) and print it along with the raw timings for both CPU and GPU.

8.  Clean up all allocated memory (both host and device) and CUDA resources.

The code uses CUDA runtime API functions for memory management and event timing, and standard C++11 utilities for CPU
timing.  Error checking is included for the CUDA calls to ensure that any issues are reported clearly.  The program is
self‑contained and can be compiled with `nvcc`:

    nvcc -std=c++11 vector_add_speedup.cu -o vector_add_speedup

Running the resulting executable will output something like:

    CPU time: 12.34 ms
    GPU time:  0.56 ms
    Speedup factor: 22.04

which demonstrates the GPU's advantage for this data‑parallel operation.
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(err)                                                \
    if (err != cudaSuccess) {                                          \
        std::cerr << "CUDA error (" << err << "): " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                            \
    }

// Simple CUDA kernel for element-wise vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const int N = 1 << 20; // 1,048,576 elements
    const size_t sizeBytes = N * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C_cpu(N);
    std::vector<float> h_C_gpu(N);

    // Initialize random number generator
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < N; ++i) {
        h_A[i] = dist(rng);
        h_B[i] = dist(rng);
    }

    // ---------- CPU Timing ----------
    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    double cpu_time_ms = cpu_duration.count();

    // ---------- GPU Timing ----------
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sizeBytes));
    CUDA_CHECK(cudaMalloc(&d_B, sizeBytes));
    CUDA_CHECK(cudaMalloc(&d_C, sizeBytes));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeBytes, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Record start event
    CUDA_CHECK(cudaEventRecord(startEvent, 0));

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stopEvent, 0));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    // Calculate elapsed time
    float gpu_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, startEvent, stopEvent));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, sizeBytes, cudaMemcpyDeviceToHost));

    // Clean up device memory and events
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    // ---------- Speedup Calculation ----------
    double speedup = cpu_time_ms / gpu_time_ms;

    // ---------- Output ----------
    std::cout << "CPU time: " << cpu_time_ms << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Speedup factor (CPU/GPU): " << speedup << std::endl;

    // Optional validation: compare CPU and GPU results
    /*
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-5) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": CPU=" << h_C_cpu[i]
                      << ", GPU=" << h_C_gpu[i] << std::endl;
            break;
        }
    }
    std::cout << "Validation " << (correct ? "PASSED" : "FAILED") << std::endl;
    */

    return 0;
}
