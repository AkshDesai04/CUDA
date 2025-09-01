/*
Aim: Query and print the `concurrentKernels` property.

Thoughts:
- We'll query the number of CUDA devices with cudaGetDeviceCount.
- For each device, retrieve its properties via cudaGetDeviceProperties.
- The `concurrentKernels` field indicates whether the device supports concurrent kernel execution.
- We'll print the device index, name, and a Yes/No indicating support.
- Error handling will be performed after each CUDA API call to catch any failures.
- The program uses the CUDA Runtime API and standard C++ I/O for simplicity.
- It can be compiled with `nvcc -o check_concurrentKernels check_concurrentKernels.cu`.
*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get properties for device " << dev << ": "
                      << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Supports concurrent kernels: "
                  << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
    }

    return EXIT_SUCCESS;
}
