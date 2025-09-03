```cpp
/*
Aim of the program: Implement `D[i] = A[i] * B[i] + C[i]` where C is a third input vector.

Thinking:
- The problem is a simple element-wise operation on three input vectors A, B, and C to produce an output vector D.
- Since CUDA is used, the operation will be parallelized across many GPU threads, each computing one element of D.
- The kernel will check that the thread index is within bounds (i < N) before performing the calculation.
- Host code will:
  1. Define vector size N.
  2. Allocate host memory for A, B, C, D.
  3. Initialize A, B, C with sample values.
  4. Allocate device memory for A, B, C, D.
  5. Copy host vectors A, B, C to device.
  6. Launch the kernel with enough blocks and threads.
  7. Copy the result vector D back to host.
  8. Optionally print a few results to verify correctness.
  9. Free device and host memory.
- A simple error checking macro (CUDA_CHECK) will be used for readability.
- The kernel uses float data type; it can be easily changed to double if needed.
*/

#include <iostream>
#include <cuda_runtime.h>

// Simple CUDA error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__   \
                      << ", code: " << err << ", reason: "                 \
                      << cudaGetErrorString(err) << std::endl;             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel: D[i] = A[i] * B[i] + C[i]
__global__ void vectorMulAdd(const float *A, const float *B, const float *C, float *D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        D[idx] = A[idx] * B[idx] + C[idx];
    }
}

int main() {
    const int N = 1 << 20;          // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    float *h_D = new float[N];

    // Initialize host vectors with sample values
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.001f;
        h_B[i] = static_cast<float>(i) * 0.002f;
        h_C[i] = static_cast<float>(i) * 0.003f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMalloc(&d_D, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorMulAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    std::cout << "Sample results:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "D[" << i << "] = " << h_D[i] << std::endl;
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_D;

    return 0;
}
```