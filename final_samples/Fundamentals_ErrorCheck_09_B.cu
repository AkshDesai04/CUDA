```cuda
/* 
Aim of the program: 
Wrap the `main` logic of your vector add program in a `try...catch` block to handle the `CudaException`. 

Thinking:
1. We need a complete CUDA vector addition example written in C++ (.cu file). 
2. Define a custom `CudaException` class that captures a CUDA error code and provides a human readable message. 
3. Use a macro `CUDA_CHECK` to wrap CUDA API calls; if a call fails, throw `CudaException`. 
4. Write a simple `vectorAdd` kernel that performs element‑wise addition of two arrays. 
5. In `main`, allocate host vectors, initialize them, allocate device memory, copy data to device, launch the kernel, synchronize, copy results back, and verify correctness. 
6. All of this logic is wrapped inside a `try` block. 
7. Catch `CudaException` specifically to print the CUDA error message, and catch any other `std::exception` as a safety net. 
8. The program should compile with `nvcc` and run on a system with CUDA installed. 
*/

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

// Custom exception to encapsulate CUDA errors
class CudaException : public std::exception {
public:
    CudaException(cudaError_t code, const char* func)
        : errorCode(code), errorFunction(func) {
        errorMessage = std::string("CUDA error in function '") + func +
                       "': " + cudaGetErrorString(code);
    }

    const char* what() const noexcept override { return errorMessage.c_str(); }

    cudaError_t code() const noexcept { return errorCode; }

private:
    cudaError_t errorCode;
    const char* errorFunction;
    std::string errorMessage;
};

// Macro to check CUDA calls and throw CudaException on error
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw CudaException(err, #call); \
    } \
} while (0)

// CUDA kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20;          // 1M elements
    const int bytes = N * sizeof(float);

    // Host vectors
    std::vector<float> h_A(N), h_B(N), h_C(N);

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Device pointers
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    try {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

        // Launch kernel
        const int threadsPerBlock = 256;
        const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());

        // Wait for GPU to finish
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

        // Verify results
        for (int i = 0; i < N; ++i) {
            float expected = h_A[i] + h_B[i];
            if (fabs(h_C[i] - expected) > 1e-5f) {
                throw std::runtime_error("Verification failed at index " + std::to_string(i));
            }
        }

        std::cout << "Vector addition successful. All values verified." << std::endl;
    }
    catch (const CudaException& e) {
        std::cerr << "CUDA Exception caught: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard exception caught: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return EXIT_SUCCESS;
}
```