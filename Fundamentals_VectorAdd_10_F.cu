/*
Make sure your CPU code is compiled with optimizations (e.g., `-O3` in GCC/Clang) for a fair comparison. Modern compilers can auto-vectorize the CPU loop using SSE/AVX instructions.

In this CUDA program we perform a simple element‑wise vector addition of two large arrays. 
The goal is to compare the performance of a straightforward CPU implementation against a CUDA kernel executing on the GPU. 
The CPU loop is intentionally written in plain C++ without any intrinsics or parallelization; the hope is that modern compilers will auto‑vectorize it when compiled with high optimization levels such as `-O3`. 
We also measure execution time on both sides using `std::chrono` for the CPU and CUDA event timers for the GPU. 
The code includes basic error checking for CUDA API calls, uses pinned memory for faster host‑to‑device copies, and frees all resources before exit.

Compile this file with something like:
    nvcc -O3 -arch=sm_70 -o vector_add vector_add.cu
or with any suitable GPU architecture flag. 
The `-O3` flag ensures that the CPU loop receives the same level of optimization as the GPU kernel, making the comparison fair. 
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            std::cerr << "CUDA error at " << __FILE__ << ":"     \
                      << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                 \
        }                                                        \
    } while (0)

// CUDA kernel for vector addition: c = a + b
__global__ void vectorAddKernel(const float *a, const float *b, float *c, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CPU implementation of vector addition
void vectorAddCPU(const float *a, const float *b, float *c, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const size_t N = 1 << 24;  // 16M elements (~64MB per array)
    const size_t bytes = N * sizeof(float);

    // Allocate pinned host memory
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    CUDA_CHECK(cudaMallocHost((void**)&h_a, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_b, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_c_cpu, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_c_gpu, bytes));

    // Initialize input data
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }

    // Device memory allocation
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // ----------------- GPU execution -----------------
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t startG, stopG;
    CUDA_CHECK(cudaEventCreate(&startG));
    CUDA_CHECK(cudaEventCreate(&stopG));

    CUDA_CHECK(cudaEventRecord(startG, 0));
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stopG, 0));
    CUDA_CHECK(cudaEventSynchronize(stopG));

    float gpuTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTimeMs, startG, stopG));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    // ----------------- CPU execution -----------------
    auto cpuStart = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_a, h_b, h_c_cpu, N);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = cpuEnd - cpuStart;

    // Verify correctness
    bool match = true;
    for (size_t i = 0; i < N; ++i) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": CPU=" << h_c_cpu[i]
                      << " GPU=" << h_c_gpu[i] << std::endl;
            match = false;
            break;
        }
    }

    std::cout << "CPU time: " << cpuDuration.count() << " ms\n";
    std::cout << "GPU time (including memory transfer): " << gpuTimeMs << " ms\n";
    std::cout << "Verification: " << (match ? "PASSED" : "FAILED") << std::endl;

    // Clean up
    CUDA_CHECK(cudaEventDestroy(startG));
    CUDA_CHECK(cudaEventDestroy(stopG));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c_cpu));
    CUDA_CHECK(cudaFreeHost(h_c_gpu));

    return 0;
}
