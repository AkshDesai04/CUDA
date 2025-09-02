```cuda
/*
Aim of the program:
Convert your `checkKernelLaunch` helper function to throw instead of exit.

Thought process:
- The original `checkKernelLaunch` likely called `cudaGetLastError()` and then used `exit()` on failure.
- Exceptions are preferable in C++ code because they allow error handling to be deferred to the caller and provide stack traces.
- We modify the helper to throw a `std::runtime_error` with a descriptive message that includes the file, line, and CUDA error string.
- A macro `CHECK_KERNEL_LAUNCH()` is provided to simplify usage.
- For demonstration, a simple kernel is launched, and the helper is invoked immediately after.
- The main function is wrapped in a try-catch block to catch and report the exception.
- All necessary headers are included: `<cuda_runtime.h>` for CUDA APIs, `<iostream>` for I/O, `<stdexcept>` for exceptions, and `<string>` for string manipulation.
- The code compiles with nvcc as a C++ source file (`.cu`), and demonstrates the new behavior.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

/* Helper function that checks the result of the last CUDA kernel launch.
   If an error occurred, it throws a std::runtime_error with detailed info. */
inline void checkKernelLaunch(const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                                 ": Kernel launch failed: " + cudaGetErrorString(err));
    }
}

/* Macro to conveniently call checkKernelLaunch with the current file and line. */
#define CHECK_KERNEL_LAUNCH() checkKernelLaunch(__FILE__, __LINE__)

/* A simple sample kernel that writes its global thread index to an output array. */
__global__ void sampleKernel(int *d_out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_out[idx] = idx;
}

int main() {
    try {
        const int N = 10;
        int *d_out = nullptr;
        cudaMalloc(&d_out, N * sizeof(int));

        // Launch the kernel
        sampleKernel<<<1, N>>>(d_out);

        // Check for launch errors; will throw if there is a problem
        CHECK_KERNEL_LAUNCH();

        // Copy results back to host and print them
        int h_out[N];
        cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i) {
            std::cout << h_out[i] << " ";
        }
        std::cout << std::endl;

        cudaFree(d_out);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```