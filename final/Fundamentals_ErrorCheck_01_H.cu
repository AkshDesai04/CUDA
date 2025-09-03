```cpp
/*
Aim of the program:
In C++, change the macro to throw a custom exception instead of exiting.

My thinking:
CUDA error handling is traditionally done with macros that print an error message and exit the program when a CUDA runtime call fails.  This is convenient for quick debugging but not suitable for applications that want to recover from errors or propagate them to higher-level logic.  In host C++ code we can throw exceptions, so the macro can be rewritten to throw a custom exception type containing the error code, error string, and the location in the source code.

To illustrate this, the program defines:
1. A custom exception class `CudaException` inheriting from `std::runtime_error`. It stores the CUDA error code, message, file name and line number.
2. A macro `CUDA_CHECK` that evaluates a CUDA runtime expression, and if the returned `cudaError_t` is not `cudaSuccess`, it throws a `CudaException` constructed with the error details.

The program also includes a small dummy kernel and a host `main()` that launches it.  After the kernel launch, `CUDA_CHECK` is used to verify the launch status.  If any CUDA error occurs, the exception will be caught in the `try`/`catch` block, and an informative error message will be printed without terminating the program via `exit`.

This example demonstrates how to replace a simple `exit`-based error macro with robust exception-based error handling in CUDA-enabled C++ code.

*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

// Custom exception for CUDA errors
class CudaException : public std::runtime_error {
public:
    CudaException(cudaError_t err, const char* file, int line)
        : std::runtime_error(buildMessage(err, file, line)),
          error_code_(err),
          file_(file),
          line_(line) {}

    cudaError_t errorCode() const { return error_code_; }
    const char* file() const { return file_; }
    int line() const { return line_; }

private:
    static std::string buildMessage(cudaError_t err, const char* file, int line) {
        std::string msg = "CUDA Error (" + std::to_string(err) + "): ";
        msg += cudaGetErrorString(err);
        msg += " at ";
        msg += file;
        msg += ":";
        msg += std::to_string(line);
        return msg;
    }

    cudaError_t error_code_;
    const char* file_;
    int line_;
};

// Macro to check CUDA runtime calls and throw CudaException on failure
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            throw CudaException(err, __FILE__, __LINE__);                        \
        }                                                                        \
    } while (0)

// Simple kernel for demonstration
__global__ void dummyKernel(int* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = idx * 2;
}

int main() {
    const int N = 256;
    int* d_data = nullptr;

    try {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));

        // Launch dummy kernel
        dim3 threadsPerBlock(64);
        dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
        dummyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data);

        // Check kernel launch status
        CUDA_CHECK(cudaGetLastError());

        // Synchronize to catch any kernel launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        // Free device memory
        CUDA_CHECK(cudaFree(d_data));

        std::cout << "Kernel executed successfully." << std::endl;
    } catch (const CudaException& e) {
        std::cerr << "Caught CudaException: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Caught std::exception: " << e.what() << std::endl;
    }

    return 0;
}
```