```cpp
/*
Does `cudaDeviceSynchronize` have a performance cost? (Yes, it stalls the CPU).
Explain when it's appropriate to use (debugging, final result needed) and when not (in a tight loop of asynchronous operations).

Thinking:
- The goal of this example is to illustrate that `cudaDeviceSynchronize()` blocks the CPU until all previously issued GPU work is finished.
- We implement a simple vector addition kernel. The host code will launch this kernel twice:
  1. With `cudaDeviceSynchronize()` immediately after launch (synchronous usage).
  2. Without a synchronization point in a tight loop, demonstrating asynchronous execution and how the CPU can do other work in parallel.
- Timing is performed with `std::chrono` on the host to show the difference in CPU wall‑clock time.
- We also use CUDA events to measure GPU execution time, which remains the same for both cases; the difference lies in the CPU time due to the block.
- The code demonstrates:
  * When to use `cudaDeviceSynchronize()` (e.g., after kernel launches when you need the results before proceeding, or during debugging to catch errors).
  * When to avoid it (inside tight loops or pipelines where you want to overlap GPU work with CPU work).
- The program prints out timings so you can see the performance cost.
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

__global__
void vectorAdd(const float* A, const float* B, float* C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const size_t N = 1 << 20;          // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N, 0.0f);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    checkCudaError(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    checkCudaError(cudaMalloc(&d_C, bytes), "cudaMalloc d_C");

    // Copy data to device (synchronous copy)
    checkCudaError(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice), "Memcpy H2D A");
    checkCudaError(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice), "Memcpy H2D B");

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // ----- Case 1: Synchronous usage (with cudaDeviceSynchronize) -----
    {
        std::cout << "=== Synchronous execution with cudaDeviceSynchronize ===" << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        // Check for launch errors
        checkCudaError(cudaGetLastError(), "Kernel launch");

        // Synchronize: blocks the CPU until the kernel finishes
        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_time = end - start;

        // Copy result back to host
        checkCudaError(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost), "Memcpy D2H");

        std::cout << "CPU wall‑clock time (including sync): " << cpu_time.count() << " s" << std::endl;
        std::cout << "First element of result: " << h_C[0] << std::endl;
    }

    // ----- Case 2: Asynchronous usage (no sync in a tight loop) -----
    {
        std::cout << "\n=== Asynchronous execution (no cudaDeviceSynchronize in loop) ===" << std::endl;

        const int num_iterations = 100;
        std::vector<float> h_D(num_iterations * N, 0.0f); // buffer for results

        auto start = std::chrono::high_resolution_clock::now();

        // Launch multiple kernels asynchronously
        for (int i = 0; i < num_iterations; ++i) {
            vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
            // We intentionally do NOT call cudaDeviceSynchronize() here.
            // The kernel launches are queued; the CPU continues immediately.
        }

        // After launching all kernels, we do a single synchronization
        checkCudaError(cudaDeviceSynchronize(), "Final cudaDeviceSynchronize");

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_time = end - start;

        // Copy result back to host (only one result needed to verify correctness)
        checkCudaError(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost), "Memcpy D2H");

        std::cout << "CPU wall‑clock time (no sync per launch): " << cpu_time.count() << " s" << std::endl;
        std::cout << "First element of result: " << h_C[0] << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```