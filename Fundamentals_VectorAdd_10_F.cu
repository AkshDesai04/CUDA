/*
Make sure your CPU code is compiled with optimizations (e.g., `-O3` in GCC/Clang) for a fair comparison. Modern compilers can auto-vectorize the CPU loop using SSE/AVX instructions.

The aim of this program is to illustrate how a simple element‑wise vector addition can be implemented both on the CPU and on the GPU, and to compare their performance under fair conditions. The CPU implementation uses a plain for‑loop that, when compiled with `-O3`, should be auto‑vectorized by modern compilers. The GPU implementation is a straightforward CUDA kernel. Both implementations work on the same input data and produce the same results, allowing a direct performance comparison.

Key design points:
1. **Data Size** – A large array (e.g., 10⁷ elements) is used to make the GPU’s parallelism visible while still allowing the CPU to finish in a reasonable time.
2. **Timing** – CPU time is measured with `std::chrono::high_resolution_clock`. GPU time is measured using CUDA events, which capture only the kernel execution time, excluding memory transfer overheads.
3. **Verification** – After both computations, the results are compared element‑wise to ensure correctness.
4. **Memory Allocation** – Host arrays are allocated with `std::vector` for simplicity. Device arrays are allocated with `cudaMalloc`.
5. **Kernel Launch Configuration** – The block size is chosen to be 256 threads per block, a common choice that typically yields good occupancy on modern GPUs.
6. **Error Checking** – All CUDA API calls are wrapped with a simple error checking macro to surface any issues early.

By compiling this file with a compiler flag such as `-O3` for the CPU part and using `nvcc` for the GPU part, one can benchmark the relative performance. The CPU loop may be auto‑vectorized to use SIMD instructions, but the GPU kernel will still typically be faster for large vector sizes due to massive parallelism.

*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <iostream>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// GPU kernel for vector addition
__global__ void vectorAddKernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 10'000'000;          // 10 million elements
    const int bytes = N * sizeof(float);

    // Allocate host vectors
    std::vector<float> h_a(N, 1.0f);   // initialize with 1.0f
    std::vector<float> h_b(N, 2.0f);   // initialize with 2.0f
    std::vector<float> h_c_cpu(N, 0.0f);
    std::vector<float> h_c_gpu(N, 0.0f);

    // ---------- CPU computation ----------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        h_c_cpu[i] = h_a[i] + h_b[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU time: " << cpu_duration.count() << " s\n";

    // ---------- GPU computation ----------
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU kernel time: " << milliseconds / 1000.0 << " s\n";

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Clean up device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // ---------- Result verification ----------
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5f) {
            ok = false;
            std::cerr << "Mismatch at index " << i << ": CPU=" << h_c_cpu[i]
                      << " GPU=" << h_c_gpu[i] << "\n";
            break;
        }
    }
    if (ok) {
        std::cout << "Verification PASSED.\n";
    } else {
        std::cerr << "Verification FAILED.\n";
    }

    return 0;
}
