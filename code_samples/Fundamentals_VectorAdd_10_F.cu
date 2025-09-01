/*
Make sure your CPU code is compiled with optimizations (e.g., `-O3` in GCC/Clang) for a fair comparison. Modern compilers can auto-vectorize the CPU loop using SSE/AVX instructions.

Thinking:
1. We need a CUDA program that demonstrates both CPU and GPU computation of the same task so that the user can compare performance.
2. The task should be simple yet involve enough work to see a noticeable speedup on GPU: element-wise vector addition.
3. The CPU implementation will be a simple `for` loop over a large array.
4. The GPU implementation will launch a kernel that does the same addition per thread.
5. Timing: use `std::chrono` for CPU and CUDA events for GPU.
6. We must remind the user that the CPU part will be best when compiled with optimizations like `-O3` to allow auto‑vectorization.
7. Include all necessary headers, allocate memory on host and device, initialize data, run both versions, and print results.
8. The code is self‑contained and can be compiled with `nvcc -O3 -o vecadd vecadd.cu` (the `-O3` flag applies to the host compiler as well as the device compiler). 
9. No extra output besides the program source is provided, as required.
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// Size of the vectors (must be a multiple of 256 for simplicity)
#define N (1 << 24)          // 16,777,216 elements (~64 MB for float)
#define THREADS_PER_BLOCK 256

// CUDA kernel for element-wise addition
__global__ void vecAddKernel(const float* a, const float* b, float* c, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

// Helper macro to check CUDA errors
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line "   \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "."   \
                      << std::endl;                                            \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

int main()
{
    // Allocate host memory
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c_cpu(N, 0.0f);
    std::vector<float> h_c_gpu(N, 0.0f);

    // ------------------- CPU computation -------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < N; ++i)
    {
        h_c_cpu[i] = h_a[i] + h_b[i];
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU time: " << cpu_duration.count() << " s" << std::endl;

    // ------------------- GPU computation -------------------
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    size_t bytes = N * sizeof(float);

    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_c, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Launch kernel
    size_t blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    CHECK_CUDA(cudaEventRecord(start, 0));
    vecAddKernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU time (excluding data transfer): " << milliseconds / 1000.0 << " s" << std::endl;

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness
    bool correct = true;
    for (size_t i = 0; i < N; ++i)
    {
        if (h_c_gpu[i] != h_c_cpu[i])
        {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": GPU=" << h_c_gpu[i]
                      << " CPU=" << h_c_cpu[i] << std::endl;
            break;
        }
    }
    if (correct)
        std::cout << "Verification passed: GPU result matches CPU result." << std::endl;
    else
        std::cout << "Verification failed!" << std::endl;

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}
