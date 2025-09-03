/*
Check if the device supports global L1 cache (globalL1CacheSupported).

I want to write a simple CUDA C program that queries the device properties for each CUDA-capable device on the system and reports whether that device supports a global L1 cache. The CUDA Runtime API provides a struct called cudaDeviceProp that contains many fields about the device, including a boolean field named globalL1CacheSupported (available on newer GPUs). To obtain the properties I will use cudaGetDeviceProperties. I will iterate over all devices (using cudaGetDeviceCount), retrieve each device's properties, and then check the globalL1CacheSupported flag. If the flag is true I will print that the device supports a global L1 cache; otherwise I will indicate that it does not. I will also handle potential errors from the CUDA API calls and print informative messages. The program is written in standard C++ with CUDA extensions, so it should be compiled with nvcc as a .cu file.
*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return 0;
    }

    std::cout << "Found " << deviceCount << " CUDA-capable device(s)." << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed for device " << dev << ": "
                      << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "\nDevice " << dev << ": " << prop.name << std::endl;
        // Check for global L1 cache support
#if defined(CUDA_VERSION) || defined(CUDART_VERSION)
        // The globalL1CacheSupported field is available in newer CUDA releases.
        if (prop.globalL1CacheSupported) {
            std::cout << "Global L1 Cache supported: Yes" << std::endl;
        } else {
            std::cout << "Global L1 Cache supported: No" << std::endl;
        }
#else
        // If the field is not available, we cannot determine support.
        std::cout << "Global L1 Cache support information not available on this CUDA version." << std::endl;
#endif
    }

    return 0;
}
