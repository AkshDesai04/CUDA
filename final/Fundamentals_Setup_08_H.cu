/*
Aim: Print the total constant memory on the device in bytes.

Thought process:
- To determine the constant memory available on a CUDA-capable GPU, we need to query the device properties using the CUDA Runtime API.
- First, retrieve the number of CUDA devices with cudaGetDeviceCount. This ensures that at least one device exists.
- Select device 0 (or any device) for simplicity. Use cudaGetDeviceProperties to obtain a cudaDeviceProp structure that contains various device attributes.
- The field totalConstMem (or constantMemorySize in newer SDKs) holds the size of constant memory in bytes. We'll use this value for output.
- Perform basic error checking after each CUDA API call, printing descriptive messages if any call fails.
- Finally, print the constant memory size in a human-readable format (bytes).
- The program is written in standard C++ with CUDA extensions, compiled with nvcc.
*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error getting device count: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    int device = 0; // Using the first device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Error getting device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Total constant memory on device " << device << ": "
              << prop.totalConstMem << " bytes" << std::endl;

    return 0;
}
