```cpp
/*
Implement a kernel for the SAXPY (Single-precision A*X Plus Y) operation.
The operation is `Y = a*X + Y`, where `a` is a scalar float, and `X` and `Y` are vectors of floats.
The scalar `a` should be passed as an argument to the kernel.

Thinking Process:
1. Define a simple CUDA kernel that performs the SAXPY operation element-wise.
   Each thread handles one vector element.
2. Use `blockIdx.x`, `blockDim.x`, and `threadIdx.x` to compute the global index.
3. Guard against out-of-bounds accesses (`if (i < n)`).
4. Write a host function to allocate device memory, initialize host vectors,
   copy them to the device, launch the kernel, copy the result back,
   and verify correctness.
5. Include error checking via a macro to simplify debugging.
6. Use a small test case (`n = 1000`) with known expected output to demonstrate correctness.
*/

#include <iostream>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err__ = (err); \
        if (err__ != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err__) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// SAXPY kernel: Y = a*X + Y
__global__ void saxpy_kernel(int n, float a, const float* X, float* Y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Y[i] = a * X[i] + Y[i];
    }
}

int main() {
    const int n = 1000;
    const float a = 2.0f;

    // Allocate host memory
    float* h_X = new float[n];
    float* h_Y = new float[n];
    float* h_Y_ref = new float[n];

    // Initialize host vectors
    for (int i = 0; i < n; ++i) {
        h_X[i] = 0.5f * static_cast<float>(i);      // X[i] = i * 0.5
        h_Y[i] = 2.0f * static_cast<float>(i);      // Y[i] = i * 2.0
        h_Y_ref[i] = a * h_X[i] + h_Y[i];           // Reference result
    }

    // Allocate device memory
    float *d_X, *d_Y;
    CUDA_CHECK(cudaMalloc(&d_X, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y, n * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, n * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, a, d_X, d_Y);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (fabs(h_Y[i] - h_Y_ref[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": expected " << h_Y_ref[i]
                      << ", got " << h_Y[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "SAXPY kernel executed successfully and produced correct results." << std::endl;
    } else {
        std::cerr << "SAXPY kernel produced incorrect results." << std::endl;
    }

    // Clean up
    delete[] h_X;
    delete[] h_Y;
    delete[] h_Y_ref;
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));

    return 0;
}
```