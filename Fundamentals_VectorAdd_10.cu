```cpp
/* 
Aim of the program: 
Compare the performance of your CUDA vector addition kernel against a simple, single-threaded C++ loop on the CPU. Time both versions for a very large vector (e.g., 10 million elements). Remember to only time the computation, not the data transfers, for a fair comparison.

Thinking:
- The program will allocate two input arrays and two output arrays on the host. The input arrays are filled with a simple pattern (i and 2*i) to avoid any branch divergence or complex behavior.
- For the CPU version, a single for-loop will perform element-wise addition. Timing is done with std::chrono::high_resolution_clock, capturing only the loop execution time.
- For the GPU version, memory for input and output arrays is allocated on the device. Host data is copied to the device before the kernel launch; this transfer time is NOT included in the timing measurement.
- The CUDA kernel is launched with a reasonable block size (256 threads) and grid size computed to cover all elements. Timing of the kernel is performed using cudaEvent_t, which records GPU timestamps before and after the kernel. Again, data transfer time is excluded.
- After the kernel completes, the result is copied back to the host for verification; this copy time is also not part of the performance comparison.
- Finally, the program prints the elapsed times for both CPU and GPU computations and verifies that the results match.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":"       \
                      << __LINE__ << " - "                        \
                      << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                   \
        }                                                          \
    } while (0)

// CUDA kernel for vector addition
__global__ void vecAdd(const float* A, const float* B, float* C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const size_t N = 10'000'000;          // 10 million elements
    const size_t bytes = N * sizeof(float);

    // Host allocations
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C_cpu(N);
    std::vector<float> h_C_gpu(N);

    // Initialize input arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // ---------- CPU computation ----------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU time: " << cpu_duration.count() << " ms\n";

    // ---------- GPU computation ----------
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device (not timed)
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
    std::cout << "GPU kernel time: " << gpu_ms << " ms\n";

    // Copy result back to host (not timed)
    CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness
    bool match = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_C_cpu[i] != h_C_gpu[i]) {
            std::cerr << "Mismatch at index " << i
                      << ": CPU=" << h_C_cpu[i]
                      << " GPU=" << h_C_gpu[i] << std::endl;
            match = false;
            break;
        }
    }
    std::cout << "Result verification: " << (match ? "PASS" : "FAIL") << std::endl;

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```