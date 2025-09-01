```cuda
/*
Aim of the program: Implement linear interpolation (lerp): C[i] = A[i] * (1.0 - t) + B[i] * t, where t is a scalar float.

Thinking:
- The core operation is element-wise linear interpolation between two arrays A and B given a scalar t.
- We implement a CUDA kernel that runs in parallel on the GPU to compute C[i] for each index i.
- The kernel uses the standard formula: C[idx] = A[idx] * (1.0f - t) + B[idx] * t.
- In the host code, we allocate and initialize sample input arrays A and B, set a scalar t, and allocate device memory.
- We copy host arrays to the device, launch the kernel with an appropriate grid and block configuration, then copy the result back.
- Finally, we print a few values of C to verify correctness.
- Error checking macros are used to catch CUDA API errors.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

// CUDA kernel performing linear interpolation
__global__ void lerp_kernel(const float* A, const float* B, float* C, float t, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * (1.0f - t) + B[idx] * t;
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << " : " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main() {
    const int n = 10;            // Number of elements
    const float t = 0.3f;        // Interpolation parameter

    // Host arrays
    float h_A[n], h_B[n], h_C[n];

    // Initialize input arrays with sample values
    for (int i = 0; i < n; ++i) {
        h_A[i] = static_cast<float>(i);          // 0, 1, 2, ...
        h_B[i] = static_cast<float>(n - i);      // 10, 9, 8, ...
    }

    // Device pointers
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, n * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice));

    // Define block and grid sizes
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel
    lerp_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, t, n);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Output results
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "A: ";
    for (int i = 0; i < n; ++i) std::cout << h_A[i] << " ";
    std::cout << "\nB: ";
    for (int i = 0; i < n; ++i) std::cout << h_B[i] << " ";
    std::cout << "\nC (interpolated with t = " << t << "): ";
    for (int i = 0; i < n; ++i) std::cout << h_C[i] << " ";
    std::cout << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```