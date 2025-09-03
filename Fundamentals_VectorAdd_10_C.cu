/*
Plot a graph of performance (elements/sec) vs. vector size (N) for both CPU and GPU. Find the crossover point where the GPU becomes faster.

This program measures the throughput (elements processed per second) for vector addition on both CPU and GPU across a range of vector sizes.
It performs the following steps for each size:
1. Allocates and initializes input vectors A and B with dummy data.
2. Runs the vector addition on the CPU, timing it with std::chrono high‑resolution clock.
3. Runs the vector addition on the GPU, timing it with CUDA events for accurate GPU measurement.
4. Computes throughput as elements_per_second = N / time_in_seconds for both CPU and GPU.
5. Stores the results in a CSV format, which can be plotted with any external tool (e.g., gnuplot, matplotlib).
6. Detects the first vector size at which the GPU throughput exceeds the CPU throughput and reports this crossover point.

Design decisions:
- Use float vectors for simplicity; memory accesses are memory‑bound, highlighting GPU advantage for large N.
- Sizes are powers of two from 1<<10 (1024) up to 1<<28 (~268M) to cover a wide range.
- Each GPU run is warmed up with a few dummy launches before timing to reduce startup overhead effects.
- Error checking is performed after each CUDA call.
- The program prints a header line for CSV compatibility and a final line indicating the crossover size.

The resulting output can be redirected to a file (e.g., `./performance > perf.csv`) and plotted externally.
The program is self‑contained and can be compiled with `nvcc -o performance performance.cu`.
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <limits>

// CUDA error checking macro
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// CPU implementation of vector addition
void cpuVectorAdd(const std::vector<float> &A, const std::vector<float> &B,
                  std::vector<float> &C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Parameters
    const int minExp = 10;          // 2^10 = 1024
    const int maxExp = 28;          // 2^28 ≈ 268M
    const int blockSize = 256;      // threads per block
    const int warmupRuns = 5;       // GPU warmup runs

    // Output CSV header
    std::cout << "N,CPU_elements_per_sec,GPU_elements_per_sec\n";

    bool crossoverFound = false;
    long long crossoverN = 0;

    for (int exp = minExp; exp <= maxExp; ++exp) {
        int N = 1 << exp;

        // Allocate host memory
        std::vector<float> h_A(N, 1.0f);
        std::vector<float> h_B(N, 2.0f);
        std::vector<float> h_C_cpu(N, 0.0f);
        std::vector<float> h_C_gpu(N, 0.0f);

        // ------- CPU timing -------
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpuVectorAdd(h_A, h_B, h_C_cpu, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
        double cpu_time_sec = cpu_duration.count();
        double cpu_throughput = N / cpu_time_sec;

        // ------- GPU timing -------
        float *d_A = nullptr;
        float *d_B = nullptr;
        float *d_C = nullptr;

        CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        // Launch parameters
        int gridSize = (N + blockSize - 1) / blockSize;

        // Warmup runs
        for (int w = 0; w < warmupRuns; ++w) {
            vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timing events
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));

        CUDA_CHECK(cudaEventSynchronize(stop));
        float gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

        double gpu_time_sec = gpu_ms / 1000.0;
        double gpu_throughput = N / gpu_time_sec;

        // Copy result back (not strictly necessary for performance)
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Free device memory
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // Output results
        std::cout << N << ',' << std::fixed << std::setprecision(2) << cpu_throughput << ','
                  << gpu_throughput << '\n';

        // Determine crossover
        if (!crossoverFound && gpu_throughput > cpu_throughput) {
            crossoverFound = true;
            crossoverN = N;
        }
    }

    if (crossoverFound) {
        std::cout << "\nCrossover point: GPU becomes faster at vector size N = " << crossoverN << std::endl;
    } else {
        std::cout << "\nNo crossover point found within the tested range." << std::endl;
    }

    return 0;
}
