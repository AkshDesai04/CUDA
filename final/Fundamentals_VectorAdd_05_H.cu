```cpp
/* 
Aim of the program: Implement `C[i] = pow(A[i], 2.5f)`.

Thinking:
- The task is to compute each element of array C as A[i] raised to the power 2.5f.
- CUDA provides the device function powf for single‑precision floating point.
- We'll allocate arrays A and C on both host and device, copy A to the device,
  launch a kernel that performs powf on each element, copy C back to host,
  and then print the results for verification.
- For simplicity, we'll use a small fixed size array (e.g., N=10) and initialize
  it with some sample values.
- We include error checking for CUDA API calls to catch issues early.
- The kernel will be very straightforward: compute the index, check bounds,
  perform powf, and store the result.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define N 10

// CUDA kernel to compute C[i] = pow(A[i], 2.5f)
__global__ void powKernel(const float* A, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = powf(A[idx], 2.5f);
    }
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error in " << __FILE__ << ":"        \
                      << __LINE__ << " - " << cudaGetErrorString(err) \
                      << std::endl;                                 \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main() {
    // Host data
    float h_A[N];
    float h_C[N];

    // Initialize host array A with sample values
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i + 1); // 1.0, 2.0, ..., N
    }

    // Device pointers
    float *d_A = nullptr, *d_C = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel: one thread per element
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    powKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "A[i] ^ 2.5 results:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "C[" << i << "] = powf(" << h_A[i] << ", 2.5f) = " << h_C[i] << std::endl;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```