/*
Check if the GPU supports `unifiedAddressing` and print a confirmation message.

Thought process:
1. The goal is to query the device properties for the unifiedAddressing flag.
2. Use cudaGetDeviceProperties to retrieve cudaDeviceProp for the first device.
3. The unifiedAddressing member indicates if the device supports unified virtual addressing.
4. Print a clear confirmation message based on the flag.
5. Handle any errors from CUDA API calls.
6. Use standard C++ I/O for printing, and include <cuda_runtime.h> for CUDA functions.
7. Keep the program minimal, self-contained, and compilable as a .cu file.
*/

#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return EXIT_FAILURE;
    }

    // Use the first device
    int dev = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Device " << dev << ": " << prop.name << std::endl;
    if (prop.unifiedAddressing) {
        std::cout << "Unified virtual addressing is supported." << std::endl;
    } else {
        std::cout << "Unified virtual addressing is NOT supported." << std::endl;
    }

    return EXIT_SUCCESS;
}
