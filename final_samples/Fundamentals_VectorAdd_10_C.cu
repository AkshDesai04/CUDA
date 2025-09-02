```cuda
/*
Aim: Plot a graph of performance (elements/sec) vs. vector size (N) for both CPU and GPU.
Find the crossover point where the GPU becomes faster.

Thought process:
- We will perform a simple element-wise vector addition: C = A + B.
- For a range of vector sizes N, we will time the CPU implementation (using std::chrono)
  and the GPU implementation (using cudaEvent timing).
- Throughput will be calculated as elements per second: N / elapsed_time.
- We will store the results and write them to a CSV file (performance.csv) so that the
  user can plot the graph externally (e.g., using gnuplot or Excel).
- While iterating over N, we will look for the first N where the GPU throughput exceeds
  the CPU throughput; that N is the crossover point.
- To keep memory usage reasonable, we will start at a small N (1024) and double it each
  step until we reach a maximum (e.g., 1<<24 ≈ 16M elements). This will give us a good
  spread of data points.
- Error checking for CUDA API calls is included to ensure robustness.
- The kernel is a simple addition; we launch enough blocks to cover all elements.
- We use cudaMalloc and cudaMemcpy for device memory; after each run, we free the memory.
- The program prints the crossover point and writes the CSV file.
*/

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Simple element-wise addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Parameters
    const size_t N_start = 1024;
    const size_t N_max   = 1 << 24; // 16,777,216 elements (~64 MB per array)
    const size_t N_step  = 2;      // double each time

    std::vector<size_t> Ns;
    std::vector<double> cpu_perf; // elements per second
    std::vector<double> gpu_perf;

    // Host vectors
    std::vector<float> h_A, h_B, h_C;

    // For CSV output
    std::ofstream csv_file("performance.csv");
    csv_file << "N,CPU_Perf_GHz,GPU_Perf_GHz\n";

    size_t crossover_N = 0;
    bool crossover_found = false;

    for (size_t N = N_start; N <= N_max; N *= N_step) {
        // Resize host arrays
        h_A.resize(N);
        h_B.resize(N);
        h_C.resize(N);

        // Initialize host data
        for (size_t i = 0; i < N; ++i) {
            h_A[i] = static_cast<float>(i);
            h_B[i] = static_cast<float>(2 * i);
        }

        // ----------------- CPU Timing -----------------
        auto cpu_start = std::chrono::high_resolution_clock::now();

        // CPU vector addition
        for (size_t i = 0; i < N; ++i) {
            h_C[i] = h_A[i] + h_B[i];
        }

        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
        double cpu_time_sec = cpu_duration.count();
        double cpu_throughput = static_cast<double>(N) / cpu_time_sec; // elements/sec

        // ----------------- GPU Timing -----------------
        // Device pointers
        float *d_A = nullptr;
        float *d_B = nullptr;
        float *d_C = nullptr;

        CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(float)));

        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        // CUDA events for timing
        cudaEvent_t start_event, stop_event;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));

        // Launch kernel
        const int threads_per_block = 256;
        int blocks = (int)((N + threads_per_block - 1) / threads_per_block);

        CUDA_CHECK(cudaEventRecord(start_event, 0));

        vectorAdd<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);

        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event));

        float gpu_elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_elapsed_ms, start_event, stop_event));

        double gpu_time_sec = gpu_elapsed_ms / 1000.0; // convert ms to s
        double gpu_throughput = static_cast<double>(N) / gpu_time_sec;

        // Clean up
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        // Store results
        Ns.push_back(N);
        cpu_perf.push_back(cpu_throughput);
        gpu_perf.push_back(gpu_throughput);

        // Write CSV row
        csv_file << N << ","
                 << std::fixed << std::setprecision(3) << cpu_throughput / 1e9 << ","
                 << std::fixed << std::setprecision(3) << gpu_throughput / 1e9 << "\n";

        // Check for crossover
        if (!crossover_found && gpu_throughput > cpu_throughput) {
            crossover_N = N;
            crossover_found = true;
        }

        // Print progress
        std::cout << "N = " << N
                  << " | CPU: " << std::fixed << std::setprecision(2)
                  << cpu_throughput / 1e9 << " Gelem/s"
                  << " | GPU: " << gpu_throughput / 1e9 << " Gelem/s"
                  << "\n";
    }

    csv_file.close();

    if (crossover_found) {
        std::cout << "\nCrossover point found: GPU becomes faster at N = " << crossover_N << "\n";
    } else {
        std::cout << "\nCrossover point not found within tested range.\n";
    }

    return 0;
}
```