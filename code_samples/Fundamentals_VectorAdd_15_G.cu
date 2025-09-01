/*
Aim of the program:
Fuse a copy and a scale operation: `B[i] = A[i] * scalar`.

Thinking and design notes:
- The task is to perform a fused operation on an array: copy elements from A to B while scaling them by a given scalar.
- In CUDA this is naturally expressed as a single kernel that takes pointers to A and B, the scalar, and the number of elements N.
- Each thread processes one or more elements using a grid‑stride loop so that the kernel works for any size of N and any launch configuration.
- Host code:
  * Define N, allocate host arrays A_h and B_h.
  * Initialize A_h with some sample values (e.g., i+1.0f).
  * Allocate device memory for A_d and B_d.
  * Copy A_h to A_d.
  * Launch the kernel with a chosen block size (e.g., 256 threads) and enough blocks to cover N.
  * Copy B_d back to B_h.
  * Verify results (print first few elements).
  * Free device and host memory.
- Error checking is performed after CUDA API calls using a helper macro `CUDA_CHECK` that prints the error string and exits if a call fails.
- The kernel is simple: each thread computes the destination index and writes `B[i] = A[i] * scalar`. The loop ensures all elements are covered even when N > blockDim.x * gridDim.x.
- The program is written in standard C with CUDA extensions, and can be compiled with `nvcc`. It includes necessary headers (`cuda_runtime.h`, `iostream`, `cstdlib`) and uses `std::cout` for output.

*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Macro for error checking
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line "      \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "."      \
                      << std::endl;                                              \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

// Kernel: fused copy and scale
__global__ void scaleKernel(const float* __restrict__ A, float* __restrict__ B,
                            float scalar, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        B[i] = A[i] * scalar;
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const float scalar = 2.5f;

    // Host memory allocation
    float* h_A = new float[N];
    float* h_B = new float[N];

    // Initialize host array A
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) + 1.0f; // e.g., 1.0, 2.0, ...
    }

    // Device memory allocation
    float *d_A = nullptr, *d_B = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the fused kernel
    scaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, scalar, N);

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 elements
    std::cout << "First 10 elements of B (A * scalar):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "B[" << i << "] = " << h_B[i] << std::endl;
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    std::cout << "Done." << std::endl;
    return 0;
}
