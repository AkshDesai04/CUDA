/*
Check if the device supports host-mapped pageable memory access (`pageableMemoryAccess`).

Thought process:
1. The goal is to query the CUDA device properties to determine whether each GPU
   supports pageable memory mapping. The property is `pageableMemoryAccess` in the
   `cudaDeviceProp` struct.
2. Use `cudaGetDeviceCount` to find the number of devices.
3. For each device:
   a. Call `cudaGetDeviceProperties` to fill a `cudaDeviceProp` struct.
   b. Inspect the `pageableMemoryAccess` boolean field.
   c. Print a clear message indicating support or lack thereof.
4. Perform error checking after each CUDA API call, printing an error and
   returning a non‑zero exit code if something fails.
5. The program compiles with `nvcc` and runs as a simple console application.

The resulting file is a single .cu file with no external dependencies beyond the
CUDA runtime.
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
        std::cout << "No CUDA-capable devices found." << std::endl;
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "Error getting properties for device " << dev
                      << ": " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << dev << " (" << prop.name << "): "
                  << (prop.pageableMemoryAccess ? "Supports" : "Does NOT support")
                  << " host-mapped pageable memory access (pageableMemoryAccess)." << std::endl;
    }

    return 0;
}
