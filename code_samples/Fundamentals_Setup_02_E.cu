```cpp
/*
Write a program that finds the device with the most global memory and prints its name and index.

Thinking process:
- Include CUDA runtime API and standard IO.
- Determine how many CUDA-capable devices are present using cudaGetDeviceCount.
- For each device, retrieve properties with cudaGetDeviceProperties.
- Track the device that has the maximum globalMemory value.
- After iterating, print the index and name of that device.
- Handle any CUDA errors appropriately, printing error messages and exiting if necessary.
- The program is intended to be compiled with nvcc and run on a system with CUDA installed.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return EXIT_FAILURE;
    }

    int maxIndex = -1;
    size_t maxMemory = 0;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed for device " << dev << ": "
                      << cudaGetErrorString(err) << std::endl;
            return EXIT_FAILURE;
        }

        if (prop.totalGlobalMem > maxMemory) {
            maxMemory = prop.totalGlobalMem;
            maxIndex = dev;
        }
    }

    if (maxIndex >= 0) {
        cudaDeviceProp bestProp;
        cudaGetDeviceProperties(&bestProp, maxIndex);
        std::cout << "Device with most global memory: "
                  << "Index " << maxIndex
                  << ", Name: " << bestProp.name
                  << ", Global Memory: " << maxMemory / (1024 * 1024) << " MB" << std::endl;
    } else {
        std::cerr << "Could not determine the device with the most global memory." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```