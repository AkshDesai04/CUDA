```cpp
/*
Implement C[i] = B[i] - A[i].
Aim: Implement C[i] = B[i] - A[i].
I considered a simple CUDA program that demonstrates vector subtraction.
I decided to use float arrays for simplicity, but any numeric type would work.
The program allocates host arrays, initializes them with example values,
allocates device memory, copies data to the device, launches a kernel that
computes C[i] = B[i] - A[i] for each element, copies the result back,
and then prints a few elements to verify correctness.
I added basic CUDA error checking using a macro for clarity.
I chose a block size of 256 and calculated the number of blocks needed.
This code is self‑contained and compiles with nvcc into a .cu file.
*/

#include <iostream>
#include <cuda_runtime.h>

// Simple macro for CUDA error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                      << " : " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel to perform C[i] = B[i] - A[i]
__global__ void vecSubtract(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = B[idx] - A[idx];
    }
}

int main() {
    const int N = 1024;                  // Number of elements
    const int size = N * sizeof(float);  // Size in bytes

    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize host arrays with sample data
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);          // 0, 1, 2, ...
        h_B[i] = static_cast<float>(2 * i);      // 0, 2, 4, ...
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecSubtract<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Print first 10 results for verification
    std::cout << "First 10 results of C[i] = B[i] - A[i]:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```