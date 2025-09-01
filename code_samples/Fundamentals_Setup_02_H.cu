/*
Aim: For each device, print its L2 cache size in megabytes (MB).

Thinking:
- To retrieve information about CUDA-capable devices, use the CUDA Runtime API.
- `cudaGetDeviceCount` gives the number of devices.
- For each device, call `cudaGetDeviceProperties` to obtain a `cudaDeviceProp` struct.
- The struct contains `l2CacheSize`, which holds the size of the L2 cache in bytes.
- Convert bytes to megabytes: MB = bytes / (1024 * 1024).
- Output the device index and L2 cache size formatted to two decimal places.
- Include basic error checking for CUDA API calls.
- Use standard C++ headers for output formatting.
*/

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error getting device count: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return 0;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "Error getting properties for device " << dev << ": "
                      << cudaGetErrorString(err) << std::endl;
            continue;
        }

        // l2CacheSize is in bytes
        double l2SizeMB = static_cast<double>(prop.l2CacheSize) / (1024.0 * 1024.0);
        std::cout << "Device " << dev << " (" << prop.name << "): "
                  << std::fixed << std::setprecision(2)
                  << l2SizeMB << " MB L2 cache" << std::endl;
    }

    return 0;
}
