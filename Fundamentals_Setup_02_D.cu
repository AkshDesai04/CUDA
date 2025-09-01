/*
Iterate through all devices and print the maximum number of threads per block for each.

Thinking:
- Use the CUDA Runtime API to query the number of available devices with cudaGetDeviceCount().
- For each device index, call cudaGetDeviceProperties() to obtain a cudaDeviceProp struct.
- The maximum number of threads per block is stored in the field maxThreadsPerBlock.
- Print each device's index and the corresponding maximum threads per block.
- Add basic error checking: if any CUDA API call fails, output the error string via cudaGetErrorString() and exit.
- The program is simple, uses standard C++ I/O, and can be compiled with nvcc.
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
        std::cout << "No CUDA-capable devices found." << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get properties for device " << dev << ": "
                      << cudaGetErrorString(err) << std::endl;
            continue;
        }
        std::cout << "Device " << dev << " (" << prop.name << "): "
                  << "maxThreadsPerBlock = " << prop.maxThreadsPerBlock << std::endl;
    }

    return EXIT_SUCCESS;
}
