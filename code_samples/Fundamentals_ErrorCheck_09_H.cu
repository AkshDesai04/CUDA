/*
Aim: Create a `CHECK_CUDA_THROW` macro for your new exception-based error handling.

Thinking:
- CUDA API functions typically return a `cudaError_t`. We want a macro that checks the return value
  and throws a C++ exception if it indicates an error.
- The macro should be safe to use in any expression context, so we wrap it in `do { ... } while(0)`.
- For clarity and code reuse, we define an inline helper function `checkCudaError` that receives the
  error code, file, and line number, and throws a `std::runtime_error` with a descriptive message.
- The macro simply captures the return value of the CUDA call, passes it to the helper, and expands
  to nothing if the call succeeds.
- We include a minimal example kernel and `main` function that demonstrates the macro in action.
  The example performs a CUDA memory allocation and a kernel launch, each wrapped with the macro.
  Any failure will be caught by the surrounding `try/catch` block.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

// Helper function to throw an exception when a CUDA error occurs
inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::string message = std::string("CUDA error: ") + cudaGetErrorString(err);
        message += " in file " + std::string(file) + " at line " + std::to_string(line);
        throw std::runtime_error(message);
    }
}

// Macro that checks the result of a CUDA API call and throws an exception on failure
#define CHECK_CUDA_THROW(call) \
    do { \
        cudaError_t err = (call); \
        checkCudaError(err, __FILE__, __LINE__); \
    } while(0)

// Simple kernel to demonstrate usage
__global__ void dummyKernel() {
    // No-op
}

int main() {
    try {
        // Allocate device memory
        float* d_ptr = nullptr;
        size_t size = 10 * sizeof(float);
        CHECK_CUDA_THROW(cudaMalloc(&d_ptr, size));

        // Launch kernel
        dummyKernel<<<1, 1>>>();
        // Check for launch errors
        CHECK_CUDA_THROW(cudaGetLastError());

        // Clean up
        CHECK_CUDA_THROW(cudaFree(d_ptr));

        std::cout << "CUDA operations completed successfully." << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Caught CUDA exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
