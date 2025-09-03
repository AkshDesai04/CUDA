```cpp
/*
Aim of the program:
Nest `try...catch` blocks to show how different levels of an application could handle CUDA errors differently.

Thinking:
The goal is to demonstrate how CUDA error handling can be layered in a C++ application by using nested
try/catch blocks.  
We define a small CUDA kernel that multiplies each element of an array by 2.  
Memory allocation and kernel launch are wrapped in separate helper functions that throw a custom
`CudaException` when a CUDA API call fails.  
Each helper function adds context to the exception and re‑throws it, allowing callers to decide
how to react.  
A `compute` function groups the two helpers inside its own try/catch, prints a higher‑level
message, and rethrows the exception to propagate the error to the caller.  
The `main` function calls `compute` inside an outer try/catch that catches all exceptions,
logs a final error message, and performs cleanup.  

This structure shows three levels of error handling:
1. Low‑level helpers – catch CUDA API errors and add context.
2. Intermediate `compute` – catch, log, and propagate.
3. High‑level `main` – catch, log, and exit gracefully.

*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>

// Kernel that multiplies each element by 2
__global__ void multiplyKernel(float* d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_arr[idx] *= 2.0f;
}

// Custom exception type for CUDA errors
class CudaException : public std::runtime_error {
public:
    cudaError_t errorCode;
    CudaException(const std::string& msg, cudaError_t err)
        : std::runtime_error(msg), errorCode(err) {}
};

// Helper to allocate and initialize device memory
void allocateMemory(float*& d_arr, int size) {
    try {
        cudaError_t err = cudaMalloc((void**)&d_arr, size * sizeof(float));
        if (err != cudaSuccess) {
            throw CudaException("cudaMalloc failed", err);
        }

        err = cudaMemset(d_arr, 0, size * sizeof(float));
        if (err != cudaSuccess) {
            throw CudaException("cudaMemset failed", err);
        }
    } catch (const CudaException& e) {
        throw CudaException("allocateMemory: " + std::string(e.what()), e.errorCode);
    }
}

// Helper to launch the kernel
void launchKernel(float* d_arr, int size) {
    try {
        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);

        multiplyKernel<<<grid, block>>>(d_arr, size);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw CudaException("Kernel launch failed", err);
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw CudaException("Kernel sync failed", err);
        }
    } catch (const CudaException& e) {
        throw CudaException("launchKernel: " + std::string(e.what()), e.errorCode);
    }
}

// Intermediate level that groups operations
void compute() {
    try {
        int size = 1 << 20;          // 1M elements
        float* d_arr = nullptr;

        allocateMemory(d_arr, size);
        launchKernel(d_arr, size);

        // Copy back to host and verify first element
        std::vector<float> h_arr(size);
        cudaError_t err = cudaMemcpy(h_arr.data(), d_arr, size * sizeof(float),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw CudaException("cudaMemcpy failed", err);
        }

        std::cout << "First element after kernel: " << h_arr[0] << std::endl;

        cudaFree(d_arr);
    } catch (const CudaException& e) {
        std::cerr << "Compute level error: " << e.what()
                  << " (" << cudaGetErrorString(e.errorCode) << ")" << std::endl;
        throw;  // propagate to outer level
    }
}

int main() {
    try {
        compute();
    } catch (const CudaException& e) {
        std::cerr << "Main level error: " << e.what()
                  << " (" << cudaGetErrorString(e.errorCode) << ")" << std::endl;
        cudaDeviceReset();
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        cudaDeviceReset();
        return EXIT_FAILURE;
    }

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
```