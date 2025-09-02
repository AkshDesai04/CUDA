```cpp
/*
Perform the comparison on a machine with a high-end CPU and a low-end GPU, and vice-versa. Observe how the results change.

Thought Process:
1. The goal is to compare the performance of a simple computational task when executed on a CPU and on a GPU.
2. Since the program cannot detect whether the GPU is “high-end” or “low-end”, it simply measures the time taken by the same operation on both devices and prints the results.
3. We will implement two kernels:
   - A memory‑bound kernel (vector addition) which mainly stresses memory bandwidth.
   - A compute‑bound kernel (a dummy heavy arithmetic loop) which stresses the arithmetic units.
4. For each kernel we will run the operation on:
   - The CPU (single‑threaded) using std::chrono for timing.
   - The GPU using CUDA events for accurate timing.
5. We allocate large arrays (e.g., 256M elements) to make the differences noticeable on both a powerful CPU and a weaker GPU.
6. The user can compile with `nvcc -O2 -o compare compare.cu` and run on any system with CUDA support.
7. The output will show the elapsed time for each version; by running the program on different hardware setups, the user can observe how the relative performance changes.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cassert>

// Size of the test arrays (256 million floats ~ 1 GB)
const size_t N = 256 * 1024 * 1024;

// CUDA kernel for vector addition: C = A + B
__global__ void vecAddKernel(const float *A, const float *B, float *C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA kernel for compute‑heavy operation: apply many arithmetic operations per element
__global__ void heavyComputeKernel(const float *A, float *C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = A[idx];
        // Perform 1024 arithmetic operations to stress ALUs
        for (int i = 0; i < 1024; ++i) {
            val = sinf(val) * cosf(val) + expf(val);
        }
        C[idx] = val;
    }
}

// Helper to check CUDA errors
inline void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// CPU implementation of vector addition
void vecAddCPU(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C) {
    for (size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// CPU implementation of heavy compute
void heavyComputeCPU(const std::vector<float> &A, std::vector<float> &C) {
    for (size_t i = 0; i < N; ++i) {
        float val = A[i];
        for (int j = 0; j < 1024; ++j) {
            val = sinf(val) * cosf(val) + expf(val);
        }
        C[i] = val;
    }
}

int main() {
    // Allocate host data
    std::vector<float> h_A(N), h_B(N), h_C(N);

    // Initialize input arrays with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = dist(rng);
        h_B[i] = dist(rng);
    }

    // ---------- GPU Vector Addition ----------
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCudaError(cudaMalloc(&d_A, N * sizeof(float)), "cudaMalloc d_A");
    checkCudaError(cudaMalloc(&d_B, N * sizeof(float)), "cudaMalloc d_B");
    checkCudaError(cudaMalloc(&d_C, N * sizeof(float)), "cudaMalloc d_C");

    // Copy inputs to device
    checkCudaError(cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D A");
    checkCudaError(cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D B");

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Define grid/block
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel and time
    checkCudaError(cudaEventRecord(start, 0), "cudaEventRecord start");
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    checkCudaError(cudaEventRecord(stop, 0), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float gpuVecAddTimeMs = 0.0f;
    checkCudaError(cudaEventElapsedTime(&gpuVecAddTimeMs, start, stop), "cudaEventElapsedTime");

    // Copy result back
    checkCudaError(cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H C");

    // Clean up
    checkCudaError(cudaFree(d_A), "cudaFree d_A");
    checkCudaError(cudaFree(d_B), "cudaFree d_B");
    checkCudaError(cudaFree(d_C), "cudaFree d_C");
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");

    // CPU vector addition timing
    auto cpuStart = std::chrono::high_resolution_clock::now();
    vecAddCPU(h_A, h_B, h_C);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuVecAddDuration = cpuEnd - cpuStart;

    // ---------- GPU Heavy Compute ----------
    // Allocate device memory again
    checkCudaError(cudaMalloc(&d_A, N * sizeof(float)), "cudaMalloc d_A");
    checkCudaError(cudaMalloc(&d_C, N * sizeof(float)), "cudaMalloc d_C");

    // Copy input to device
    checkCudaError(cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D A");

    // Timing
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    checkCudaError(cudaEventRecord(start, 0), "cudaEventRecord start");
    heavyComputeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);
    checkCudaError(cudaEventRecord(stop, 0), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float gpuHeavyTimeMs = 0.0f;
    checkCudaError(cudaEventElapsedTime(&gpuHeavyTimeMs, start, stop), "cudaEventElapsedTime");

    // Copy result back
    checkCudaError(cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H C");

    // Clean up
    checkCudaError(cudaFree(d_A), "cudaFree d_A");
    checkCudaError(cudaFree(d_C), "cudaFree d_C");
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");

    // CPU heavy compute timing
    auto cpuStartHeavy = std::chrono::high_resolution_clock::now();
    heavyComputeCPU(h_A, h_C);
    auto cpuEndHeavy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuHeavyDuration = cpuEndHeavy - cpuStartHeavy;

    // ---------- Results ----------
    std::cout << "=== Performance Comparison ===" << std::endl;
    std::cout << "Vector Addition:" << std::endl;
    std::cout << "  CPU time (ms):        " << cpuVecAddDuration.count() << std::endl;
    std::cout << "  GPU time (ms):        " << gpuVecAddTimeMs << std::endl;
    std::cout << std::endl;

    std::cout << "Heavy Compute (1024 sin/cos/exp ops per element):" << std::endl;
    std::cout << "  CPU time (ms):        " << cpuHeavyDuration.count() << std::endl;
    std::cout << "  GPU time (ms):        " << gpuHeavyTimeMs << std::endl;
    std::cout << std::endl;

    std::cout << "Interpretation:" << std::endl;
    std::cout << "  - On a high‑end CPU with a low‑end GPU, you will typically see the GPU slower for the simple memory‑bound vector addition but may still outperform for the compute‑heavy kernel, depending on GPU compute cores." << std::endl;
    std::cout << "  - On a low‑end CPU with a high‑end GPU, the GPU will usually dominate both kernels, especially for large data sizes." << std::endl;
    std::cout << "  - Observe the ratio of CPU/GPU times to understand the hardware impact." << std::endl;

    return 0;
}
```