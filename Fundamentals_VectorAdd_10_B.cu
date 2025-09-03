```cuda
/* 
   Aim of the program: Write a multi-threaded CPU version of vector addition (e.g., using OpenMP) and compare it to the GPU version.

   Thought process:
   1. The goal is to provide two implementations of a simple element-wise addition of two large vectors:
        c[i] = a[i] + b[i]
      One implementation will use the CPU with OpenMP to parallelize across the CPU cores.
      The second implementation will run on the GPU using a CUDA kernel.
   2. We need to allocate a large amount of data to see meaningful differences in performance.
      A typical choice is 10^7 or 2^24 (~16 million) elements. For floats, that is about 64 MB per array.
   3. The program will:
      - Allocate host arrays `a`, `b`, and `c_cpu` (CPU result) and `c_gpu` (GPU result).
      - Initialize `a` and `b` with random floats.
      - Run the CPU addition using an OpenMP parallel for loop.
      - Run the GPU addition:
        * Copy `a` and `b` to device memory.
        * Launch a kernel that writes to `c_gpu` on the device.
        * Copy `c_gpu` back to host.
      - Measure execution time of each implementation:
        * For CPU, use C++ chrono or `omp_get_wtime`.
        * For GPU, use CUDA events for accurate GPU timing.
      - Verify correctness by comparing the CPU and GPU results element-wise within a tolerance.
      - Print out the timings and a verdict on speedup and correctness.
   4. The code is written in a single .cu file so it can be compiled with NVCC.  OpenMP support is
      enabled via the `-Xcompiler -fopenmp` flag when compiling.
   5. CUDA error checking is done through a helper macro `CUDA_CHECK` that aborts on failure.
   6. The implementation is intentionally straightforward and avoids overly complex optimizations
      (e.g., shared memory tiling) so that the focus remains on the comparison between CPU and GPU.
   7. The program can be easily extended by changing the vector size or by adding more kernels.
*/

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// CUDA kernel for vector addition
__global__ void gpu_vector_add(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c,
                               size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// CPU implementation using OpenMP
void cpu_vector_add(const float* a, const float* b, float* c, size_t N)
{
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }
}

// Verify that two vectors are approximately equal
bool verify_result(const std::vector<float>& ref,
                   const std::vector<float>& test,
                   float epsilon = 1e-5f)
{
    if (ref.size() != test.size())
        return false;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::fabs(ref[i] - test[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i
                      << ": ref=" << ref[i]
                      << ", test=" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    // Vector size: 16 million elements (~64 MB per array)
    const size_t N = 1 << 24;  // 16,777,216

    std::cout << "Vector size: " << N << " elements\n";

    // Allocate host memory
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c_cpu(N);
    std::vector<float> h_c_gpu(N);

    // Initialize input vectors with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = dist(rng);
        h_b[i] = dist(rng);
    }

    // CPU (OpenMP) execution
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_vector_add(h_a.data(), h_b.data(), h_c_cpu.data(), N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU (OpenMP) time: " << cpu_duration.count() << " s\n";

    // GPU execution
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // CUDA timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    gpu_vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    std::cout << "GPU time: " << gpu_time_ms / 1000.0f << " s\n";

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Verify correctness
    bool correct = verify_result(h_c_cpu, h_c_gpu);
    std::cout << "Result verification: " << (correct ? "PASS" : "FAIL") << std::endl;

    // Compute speedup
    double speedup = cpu_duration.count() / (gpu_time_ms / 1000.0f);
    std::cout << "Speedup (CPU/GPU): " << speedup << "x\n";

    return 0;
}
```