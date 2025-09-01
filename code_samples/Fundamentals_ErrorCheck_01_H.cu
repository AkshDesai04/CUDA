/*
In C++, change the macro to throw a custom exception instead of exiting.

Thinking:
- The user wants to see how a typical CUDA error-checking macro can be modified
  to throw a C++ exception instead of calling `exit()` or terminating the
  program.
- I will create a simple CUDA program that launches a trivial kernel.
- A custom exception type `CudaException` will be defined to hold the error
  code and message.
- The macro `CUDA_CHECK` will call a helper function that checks the return
  value of a CUDA runtime API call. If the return value is not `cudaSuccess`,
  the helper will throw `CudaException`.
- The code will be self‑contained, compile with `nvcc`, and demonstrate the
  macro in action.
- All logic is in a single `.cu` file, with a multiline comment at the
  beginning containing the aim and the reasoning.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

/* Custom exception to encapsulate CUDA errors */
class CudaException : public std::runtime_error {
public:
    CudaException(cudaError_t err, const std::string& file, int line)
        : std::runtime_error(buildMessage(err, file, line)),
          errorCode(err) {}

    cudaError_t code() const { return errorCode; }

private:
    cudaError_t errorCode;

    static std::string buildMessage(cudaError_t err, const std::string& file, int line) {
        std::string msg = "CUDA Error: " + std::string(cudaGetErrorString(err));
        msg += " at " + file + ":" + std::to_string(line);
        return msg;
    }
};

/* Helper function that throws if CUDA call fails */
inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw CudaException(err, file, line);
    }
}

/* Macro that wraps CUDA runtime calls */
#define CUDA_CHECK(call) checkCudaError((call), __FILE__, __LINE__)

/* Simple kernel that does nothing */
__global__ void dummyKernel(int *d_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_data[idx] = idx;
}

int main() {
    try {
        const int N = 256;
        int *d_ptr = nullptr;
        // Allocate device memory
        CUDA_CHECK(cudaMalloc((void**)&d_ptr, N * sizeof(int)));

        // Launch kernel
        dim3 block(64);
        dim3 grid((N + block.x - 1) / block.x);
        dummyKernel<<<grid, block>>>(d_ptr);
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());

        // Copy result back (for completeness)
        int h_data[N];
        CUDA_CHECK(cudaMemcpy(h_data, d_ptr, N * sizeof(int), cudaMemcpyDeviceToHost));

        // Free device memory
        CUDA_CHECK(cudaFree(d_ptr));

        std::cout << "Kernel executed successfully. First element: " << h_data[0] << std::endl;
    }
    catch (const CudaException& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
