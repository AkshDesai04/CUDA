/*
In the `catch` block, print the detailed error message from the exception object.

Thinking:
- CUDA runtime functions return error codes rather than throwing C++ exceptions.  
- To get a C++ exception with a detailed message, wrap each CUDA call in a helper macro that checks the return status and throws `std::runtime_error` when an error occurs.  
- The exception message should include the file, line, and the CUDA error string obtained via `cudaGetErrorString`.  
- In `main`, perform a CUDA operation that is guaranteed to fail (e.g., setting an invalid device ID) so that the exception is thrown.  
- Catch the exception with a `try‑catch` block, and in the `catch` block print `e.what()`, which contains the detailed error message.  
- The program will exit with `EXIT_FAILURE` after printing the message; otherwise it prints “Success!”.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cstdlib>

// Macro to check CUDA runtime calls and throw a C++ exception on failure
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                  \
            std::ostringstream oss;                                \
            oss << "CUDA error at " << __FILE__ << ":"             \
                << __LINE__ << " - " << cudaGetErrorString(err);   \
            throw std::runtime_error(oss.str());                    \
        }                                                           \
    } while (0)

// Simple kernel that does nothing
__global__ void dummyKernel() {
    // Intentionally left blank
}

int main() {
    try {
        // Intentionally set an invalid device ID to trigger a CUDA error
        int invalidDevice = -1;
        CHECK_CUDA(cudaSetDevice(invalidDevice));

        // If the above succeeds (unlikely), launch the kernel
        dummyKernel<<<1, 1>>>();
        CHECK_CUDA(cudaGetLastError());        // Check for kernel launch errors
        CHECK_CUDA(cudaDeviceSynchronize());   // Ensure kernel completion
    } catch (const std::exception &e) {
        // Print the detailed error message from the exception object
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Success!" << std::endl;
    return EXIT_SUCCESS;
}
