```cuda
/*
Write a program that attempts to set the device to an invalid index (e.g., 999) and use proper error checking to report that the device does not exist.

Thought process:
1. The program must be a standalone .cu file that can be compiled with nvcc.
2. It should query the number of available CUDA devices, attempt to set an invalid device index (999), and handle the resulting error gracefully.
3. We will use CUDA runtime API functions: cudaGetDeviceCount, cudaSetDevice, cudaGetLastError, and cudaGetErrorString.
4. To provide clear output, we will print the total device count, attempt to set device 999, and if an error occurs, print the error code and description.
5. The error checking will be implemented in a helper macro/function that checks the return value of CUDA API calls and prints descriptive messages.
6. No GPU kernels or memory allocations are required for this simple error-reporting demonstration.
7. The program will return 0 on success (though we expect an error on setting the device), or a non-zero value if any CUDA call fails.

The final code below follows this design and should compile with a standard CUDA toolkit.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

// Helper macro to check CUDA errors
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                      << " code=" << err << " (" << cudaGetErrorString(err) \
                      << ")" << std::endl;                               \
            return EXIT_FAILURE;                                          \
        }                                                                  \
    } while (0)

int main()
{
    int deviceCount = 0;
    // Get the number of CUDA-capable devices
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    std::cout << "Number of CUDA devices available: " << deviceCount << std::endl;

    // Attempt to set an invalid device index (e.g., 999)
    int invalidDeviceIndex = 999;
    std::cout << "Attempting to set device to index " << invalidDeviceIndex << "..." << std::endl;

    cudaError_t err = cudaSetDevice(invalidDeviceIndex);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set device " << invalidDeviceIndex << ": "
                  << cudaGetErrorString(err) << std::endl;
        // Optionally, you can query the last error again
        cudaError_t lastErr = cudaGetLastError();
        if (lastErr != cudaSuccess) {
            std::cerr << "cudaGetLastError reports: " << cudaGetErrorString(lastErr) << std::endl;
        }
    } else {
        std::cout << "Successfully set device to " << invalidDeviceIndex << std::endl;
    }

    return EXIT_SUCCESS;
}
```