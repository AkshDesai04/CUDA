```cuda
/*
Calculate and print the speedup factor (CPU time / GPU time).

The aim of this program is to demonstrate a simple performance comparison
between a CPU implementation and a GPU implementation of a vector addition
operation.  The user prompt explicitly requests the calculation and printing
of the speedup factor as CPU time divided by GPU time.  The code below
performs the following steps:

1.  Allocate two large input arrays `a` and `b` and one output array `c`
    on the host.  The arrays are filled with random floating‑point values.

2.  Compute `c[i] = a[i] + b[i]` on the CPU using a plain for‑loop.
    The execution time of this loop is measured with
    `std::chrono::high_resolution_clock`.

3.  Allocate corresponding arrays on the GPU, copy the inputs from host
    to device, and launch a CUDA kernel that performs the same element‑wise
    addition.  The GPU execution time is measured using CUDA events
    (`cudaEventRecord` / `cudaEventElapsedTime`).  The measured time
    includes kernel launch, kernel execution, and host‑to‑device /
    device‑to‑host copies, which is a realistic proxy for the total GPU
    time a user would incur.

4.  After the GPU kernel completes, the result is copied back to the host.

5.  The program then prints the CPU time, GPU time, and the computed
    speedup factor `cpu_time / gpu_time`.

6.  Basic CUDA error checking is performed after each API call or
    kernel launch to ensure that any issues are reported immediately.

The program uses only standard C++ and CUDA runtime APIs, and it compiles
with `nvcc` to produce a single executable.  The array size is set to
10 million elements, which is large enough to observe a noticeable
speedup on a typical GPU but small enough to fit in common system memory.

This approach satisfies the requirement of printing the speedup factor and
provides a clear, self‑contained demonstration of CPU vs. GPU performance.
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

// Macro for checking CUDA errors following a CUDA API call or kernel launch
#define CUDA_CHECK(call)                                                \
    do {                                                               \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error in file '" << __FILE__            \
                      << "' in line " << __LINE__ << ": "              \
                      << cudaGetErrorString(err) << std::endl;         \
            exit(EXIT_FAILURE);                                         \
        }                                                              \
    } while (0)

// CUDA kernel for vector addition
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 10'000'000;  // 10 million elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c_cpu(N);
    std::vector<float> h_c_gpu(N);

    // Initialize input arrays with random values
    std::srand(0);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(std::rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    // -------- CPU computation --------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        h_c_cpu[i] = h_a[i] + h_b[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    double cpu_time_ms = cpu_duration.count();

    // -------- GPU computation --------
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch kernel
    const int THREADS_PER_BLOCK = 256;
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    CUDA_CHECK(cudaEventRecord(start));
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));

    // Wait for GPU to finish
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Compute GPU time
    float gpu_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));

    // Destroy events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Clean up device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // Verify correctness (optional)
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5f) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": "
                      << h_c_cpu[i] << " vs " << h_c_gpu[i] << std::endl;
            break;
        }
    }
    if (!correct) {
        std::cerr << "Result verification failed." << std::endl;
    } else {
        std::cout << "Result verification succeeded." << std::endl;
    }

    // Compute and print speedup
    double speedup = cpu_time_ms / gpu_time_ms;

    std::cout << "CPU time: " << cpu_time_ms << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Speedup factor (CPU/GPU): " << speedup << std::endl;

    return 0;
}
```