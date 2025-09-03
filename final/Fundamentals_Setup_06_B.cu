```cuda
/*
Aim of the program: Write a program that checks if a hypothetical block configuration of (1024, 2, 1) would be valid, given the device's limits.

Thinking process:
1. We need to retrieve device properties via cudaGetDeviceProperties.
2. The block configuration is (1024, 2, 1). We must ensure:
   a. Each dimension does not exceed the corresponding maxThreadsDim entry.
   b. The product of the three dimensions (total threads per block) does not exceed maxThreadsPerBlock.
3. We will print out the relevant device properties and whether the configuration is valid.
4. We'll support checking all devices present on the system for completeness.
5. Error handling will be minimal but sufficient: if any CUDA API call fails, we print an error message and exit.
6. The program will compile as a .cu file and can be run with nvcc.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return 1;
    }

    // Block configuration to test
    const int testBlockX = 1024;
    const int testBlockY = 2;
    const int testBlockZ = 1;
    const int totalThreads = testBlockX * testBlockY * testBlockZ;

    std::cout << "Testing block configuration (" << testBlockX << ", " << testBlockY << ", " << testBlockZ << ") on all devices." << std::endl;
    std::cout << "Total threads per block: " << totalThreads << std::endl << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed for device " << dev << ": " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Max threads per block:          " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dimension (x, y, z): (" 
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")" << std::endl;

        bool dimOk = (testBlockX <= prop.maxThreadsDim[0]) &&
                     (testBlockY <= prop.maxThreadsDim[1]) &&
                     (testBlockZ <= prop.maxThreadsDim[2]);

        bool totalOk = (totalThreads <= prop.maxThreadsPerBlock);

        if (dimOk && totalOk) {
            std::cout << "  Result: Valid configuration for this device." << std::endl;
        } else {
            std::cout << "  Result: INVALID configuration for this device." << std::endl;
            if (!dimOk) {
                std::cout << "    Reason: One or more block dimensions exceed device limits." << std::endl;
            }
            if (!totalOk) {
                std::cout << "    Reason: Total threads per block exceed device limit." << std::endl;
            }
        }
        std::cout << std::endl;
    }

    return 0;
}
```