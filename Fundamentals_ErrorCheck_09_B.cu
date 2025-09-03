```cpp
/*
Aim of the program:
Wrap the `main` logic of your vector add program in a `try...catch` block to handle the `CudaException`.

Thinking process:
1. Define a custom exception `CudaException` that stores the CUDA error string and the error code.
2. Create a helper function `checkCudaError` that takes a CUDA API call result and throws `CudaException` if it indicates failure.
3. Write a simple vector addition CUDA kernel.
4. In `main`, allocate host arrays, initialize them, allocate device memory, copy data to device, launch kernel, copy result back, and free resources.
5. Wrap all of the above logic inside a `try` block, and catch `CudaException` to report errors. Also catch any other exceptions for safety.
6. Provide clear error messages and exit with appropriate status codes.
*/

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

// Custom exception for CUDA errors
class CudaException : public std::runtime_error {
public:
    CudaException(const std::string& msg, cudaError_t err)
        : std::runtime_error(msg + ": " + cudaGetErrorString(err)), errorCode(err) {}

    cudaError_t code() const { return errorCode; }

private:
    cudaError_t errorCode;
};

// Helper to check CUDA API calls
inline void checkCudaError(cudaError_t err, const std::string& context) {
    if (err != cudaSuccess) {
        throw CudaException(context, err);
    }
}

// CUDA kernel for vector addition
__global__ void vecAddKernel(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const size_t sizeBytes = N * sizeof(float);

    // Host vectors
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c(N, 0.0f);

    // Device pointers
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    try {
        // Allocate device memory
        checkCudaError(cudaMalloc(&d_a, sizeBytes), "cudaMalloc d_a");
        checkCudaError(cudaMalloc(&d_b, sizeBytes), "cudaMalloc d_b");
        checkCudaError(cudaMalloc(&d_c, sizeBytes), "cudaMalloc d_c");

        // Copy host to device
        checkCudaError(cudaMemcpy(d_a, h_a.data(), sizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy a");
        checkCudaError(cudaMemcpy(d_b, h_b.data(), sizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy b");

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

        // Check for launch errors
        checkCudaError(cudaGetLastError(), "Kernel launch");
        // Wait for device to finish
        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel");

        // Copy result back to host
        checkCudaError(cudaMemcpy(h_c.data(), d_c, sizeBytes, cudaMemcpyDeviceToHost), "cudaMemcpy c");

        // Simple verification: print first element
        std::cout << "First element of result: " << h_c[0] << std::endl;

    } catch (const CudaException& e) {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
        // Clean up device memory if allocated
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return EXIT_FAILURE;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "Vector addition completed successfully." << std::endl;
    return EXIT_SUCCESS;
}
```