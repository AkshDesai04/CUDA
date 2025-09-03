/*
Aim of the program:
Query and print the amount of shared memory available per thread block in bytes for the primary device.

Thinking process:
1. The CUDA Runtime API provides a function `cudaGetDeviceProperties` that fills a `cudaDeviceProp` structure with device-specific information.
2. The field `sharedMemPerBlock` inside `cudaDeviceProp` contains the amount of shared memory (in bytes) that is available to each thread block on the device.
3. The "primary device" typically refers to device 0, but to be safe we can query the current device using `cudaGetDevice` and then query its properties.
4. We'll include error checking: if any CUDA call fails, we will print the error string and exit.
5. The program will print the value to standard output, e.g., "Shared memory per block: X bytes".
6. We'll write the code in C++ style (using `<iostream>`) but compile as a CUDA source file (.cu) so that it can use CUDA runtime functions.
7. No kernel launch is required; the main work is performed on the host side.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceId = 0; // primary device
    cudaError_t err = cudaGetDevice(&deviceId);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get current device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, deviceId);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Shared memory per block on device " << deviceId
              << ": " << prop.sharedMemPerBlock << " bytes" << std::endl;

    return 0;
}
