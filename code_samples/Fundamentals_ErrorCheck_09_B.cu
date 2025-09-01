```cuda
/*
Wrap the `main` logic of your vector add program in a `try...catch` block to handle the `CudaException`.

Thinking:
- The goal is to create a CUDA C++ program that performs vector addition on the GPU.
- The main computational part (memory allocation, data transfer, kernel launch, and result retrieval) should be protected by a try...catch block.
- We need a custom exception type, `CudaException`, to represent CUDA runtime errors. 
- A helper function `checkCudaError` will check CUDA API return codes and throw `CudaException` with an informative message when a failure occurs.
- In the kernel we will perform element‑wise addition of two float arrays.
- The main function will allocate host memory, initialize it, allocate device memory, copy data to the device, launch the kernel, copy the result back, free device memory, and finally free host memory. All of this will be inside a try block.
- The catch block will catch `CudaException` and print the error message, then return `EXIT_FAILURE`. If everything succeeds, `main` will return `EXIT_SUCCESS`.
- Include standard headers: `<cstdio>`, `<cstdlib>`, `<cuda_runtime.h>`, `<iostream>`, `<stdexcept>`.
- Ensure that the code compiles with `nvcc` and is self‑contained in a single .cu file.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

class CudaException : public std::runtime_error {
public:
    explicit CudaException(const std::string& message)
        : std::runtime_error(message) {}
};

// Helper macro to check CUDA errors
inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw CudaException(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// CUDA kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    if (!h_A || !h_B || !h_C) {
        std::cerr << "Host memory allocation failed." << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Device pointers
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    try {
        // Allocate device memory
        checkCudaError(cudaMalloc((void**)&d_A, size), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc((void**)&d_B, size), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc((void**)&d_C, size), "cudaMalloc d_C failed");

        // Copy data from host to device
        checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "cudaMemcpy h_A -> d_A failed");
        checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "cudaMemcpy h_B -> d_B failed");

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

        // Check for kernel launch errors
        checkCudaError(cudaGetLastError(), "Kernel launch failed");

        // Copy result back to host
        checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy d_C -> h_C failed");

        // Verify results
        bool success = true;
        for (int i = 0; i < N; ++i) {
            float expected = h_A[i] + h_B[i];
            if (fabs(h_C[i] - expected) > 1e-5) {
                std::cerr << "Verification failed at index " << i
                          << ": " << h_C[i] << " != " << expected << std::endl;
                success = false;
                break;
            }
        }
        if (success) {
            std::cout << "Vector addition successful, all results verified." << std::endl;
        } else {
            std::cerr << "Vector addition verification failed." << std::endl;
            return EXIT_FAILURE;
        }
    } catch (const CudaException& e) {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
        // Free device memory if allocated
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        // Free host memory
        free(h_A);
        free(h_B);
        free(h_C);
        return EXIT_FAILURE;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
```