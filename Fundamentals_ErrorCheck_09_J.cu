```cpp
/*
Aim of the program:
Nest `try...catch` blocks to show how different levels of an application could handle CUDA errors differently.

Thinking:
- Use C++ exceptions to propagate errors.
- Low-level function performs cudaMalloc and uses CHECK_CUDA macro.
- Intermediate function calls low-level, launches kernel, and catches exception to add context.
- Top-level main catches final exception and prints.
- Use a deliberately large allocation to trigger an error and demonstrate nested handling.
*/

#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

// Helper macro to check CUDA calls and throw exception on failure
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error: ") +              \
                                     cudaGetErrorString(err));                  \
        }                                                                       \
    } while (0)

// Simple kernel (does nothing, just a placeholder)
__global__ void dummyKernel(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 10) {
        data[idx] = idx;
    }
}

// Low-level function that allocates memory on the device
int* lowLevelAllocate(size_t size) {
    try {
        int *devPtr = nullptr;
        CHECK_CUDA(cudaMalloc((void**)&devPtr, size));
        return devPtr;
    } catch (const std::exception &e) {
        // Re-throw with additional context
        throw std::runtime_error(std::string("Low-level allocation failed: ") + e.what());
    }
}

// Intermediate function that launches kernel and synchronizes
int* intermediateKernelLaunch(size_t size) {
    try {
        int *devPtr = lowLevelAllocate(size);

        // Launch kernel (intentionally small grid/block, no real error here)
        dummyKernel<<<1, 32>>>(devPtr);
        CHECK_CUDA(cudaGetLastError());          // Check launch error
        CHECK_CUDA(cudaDeviceSynchronize());     // Wait for completion

        return devPtr;
    } catch (const std::exception &e) {
        // Add context and rethrow
        throw std::runtime_error(std::string("Intermediate level kernel launch failed: ") + e.what());
    }
}

int main() {
    try {
        // Deliberately large allocation to trigger an error (e.g., 1e10 bytes)
        size_t hugeSize = static_cast<size_t>(1e10);
        int *devPtr = intermediateKernelLaunch(hugeSize);

        // Normally, we would use devPtr here...

        // Clean up
        CHECK_CUDA(cudaFree(devPtr));
    } catch (const std::exception &e) {
        // Final error handling at application level
        std::cerr << "Top-level error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Program completed successfully." << std::endl;
    return EXIT_SUCCESS;
}
```