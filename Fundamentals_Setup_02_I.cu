```cuda
/*
 * Aim of the program (passed in the prompt word to word):
 * For each device, query and print whether it supports running kernels concurrently.
 *
 * Thinking process:
 * 1. In CUDA, each device has properties accessible via cudaGetDeviceProperties().
 *    Among these properties is `concurrentKernels`, an int that indicates whether the
 *    device can execute multiple kernels concurrently.  The value is 1 if supported,
 *    0 otherwise.
 * 2. Additionally, `asyncEngineCount` can give an indication of the number of
 *    asynchronous engines available; a value greater than zero often correlates
 *    with concurrent kernel capability, but the authoritative flag is `concurrentKernels`.
 * 3. The program will:
 *    - Query the number of devices with cudaGetDeviceCount().
 *    - For each device index, retrieve cudaDeviceProp via cudaGetDeviceProperties().
 *    - Print the device name and whether `concurrentKernels` is true or false.
 * 4. Error handling is essential: we wrap CUDA API calls with a helper macro to
 *    check return codes and abort on failure.
 * 5. Since we only need to query properties, no actual kernel launches are required.
 * 6. The code will be a standard CUDA C++ program that can be compiled with nvcc.
 *
 * The final code below implements these steps and outputs the required information.
 */

#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error in " << __FILE__             \
                      << " at line " << __LINE__ << ": "          \
                      << cudaGetErrorString(err) << std::endl;    \
            std::exit(EXIT_FAILURE);                              \
        }                                                         \
    } while (0)

int main() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return 0;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Supports concurrent kernels: "
                  << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "  Async engine count: " << prop.asyncEngineCount << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
```