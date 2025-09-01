```cpp
/*
Aim of the program: 
Compare the performance of your CUDA vector addition kernel against a simple, single-threaded C++ loop on the CPU. Time both versions for a very large vector (e.g., 10 million elements). Remember to only time the computation, not the data transfers, for a fair comparison.

Thinking process:
1. Decide vector size: 10 million elements (1e7). Use `size_t N = 10'000'000;`.
2. Allocate three host arrays: `a`, `b`, and `c_cpu` (for CPU result). Use `std::vector<float>` for ease of allocation and initialization.
3. Fill `a` and `b` with random floating-point values using `std::mt19937` and `std::uniform_real_distribution`.
4. For the GPU part:
   - Allocate device memory for `a_d`, `b_d`, and `c_d` using `cudaMalloc`.
   - Copy `a` and `b` to device using `cudaMemcpy` *before* timing; this data transfer must not be included in the kernel timing.
   - Define a simple element-wise addition kernel: each thread reads `a_d[idx] + b_d[idx]` into `c_d[idx]`. Use a block size of 256 and grid size computed from `N`.
   - Use CUDA events (`cudaEvent_t start, stop`) to time only the kernel execution: record start, launch kernel, record stop, synchronize, and then compute elapsed time in milliseconds.
   - Copy result from `c_d` back to host `c_gpu` if we want to verify correctness; this transfer is also outside the timing.
5. For the CPU part:
   - Use `std::chrono::high_resolution_clock` to measure the time of a single-loop addition: `for (size_t i=0; i<N; ++i) c_cpu[i] = a[i] + b[i];`.
   - Ensure that this timing includes only the loop execution, not any setup or memory allocation.
6. After both timings, print the elapsed times in milliseconds. Optionally verify that GPU and CPU results match within a tolerance.
7. Clean up: free device memory with `cudaFree` and destroy CUDA events.
8. The code should compile with `nvcc` and be self-contained.

Edge cases and considerations:
- Use `size_t` for indexing to avoid overflow.
- Ensure CUDA kernel launch parameters are correctly calculated: `int threadsPerBlock = 256; int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;`.
- Check CUDA API calls for errors using a helper macro for brevity.
- The random number generator should be seeded with a fixed value for reproducibility.
- Use `float` for simplicity; we could also use `double`, but `float` is faster on many GPUs.

With these steps, we get a fair performance comparison between the GPU kernel and the CPU loop, excluding data transfer overhead.
*/

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in file '" << __FILE__                  \
                      << "' in line " << __LINE__ << ": "                   \
                      << cudaGetErrorString(err) << std::endl;              \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)

// CUDA kernel for vector addition
__global__ void vecAdd(const float* a, const float* b, float* c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const size_t N = 10'000'000;            // 10 million elements
    const size_t bytes = N * sizeof(float);

    // Host vectors
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c_cpu(N);
    std::vector<float> h_c_gpu(N);

    // Random number generation
    std::mt19937 rng(12345); // fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = dist(rng);
        h_b[i] = dist(rng);
    }

    // --------- GPU part ----------
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // Copy inputs to device (not timed)
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Timing with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((N + threadsPerBlock - 1) / threadsPerBlock);

    CUDA_CHECK(cudaEventRecord(start));
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));

    // Wait for the kernel to finish
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));

    // Copy result back to host (not timed)
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Destroy events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // --------- CPU part ----------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        h_c_cpu[i] = h_a[i] + h_b[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time_ms = cpu_end - cpu_start;

    // --------- Verify correctness ----------
    bool correct = true;
    const float eps = 1e-5f;
    for (size_t i = 0; i < N; ++i) {
        if (std::fabs(h_c_cpu[i] - h_c_gpu[i]) > eps) {
            correct = false;
            std::cerr << "Mismatch at index " << i
                      << ": CPU=" << h_c_cpu[i]
                      << " GPU=" << h_c_gpu[i] << std::endl;
            break;
        }
    }

    std::cout << "Result verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "GPU time (kernel only): " << gpu_time_ms << " ms" << std::endl;
    std::cout << "CPU time (single-threaded loop): " << cpu_time_ms.count() << " ms" << std::endl;

    // Clean up device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
```