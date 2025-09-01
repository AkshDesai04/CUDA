```cpp
/*
Aim: Write a program that finds the device with the most global memory and prints its name and index.

Thinking:
1. We need to query the number of CUDA-capable devices using cudaGetDeviceCount.
2. For each device, retrieve its properties via cudaGetDeviceProperties to access the totalGlobalMem field.
3. Keep track of the device index that has the maximum global memory found so far.
4. After scanning all devices, print the name and index of the device with the largest memory.
5. Handle error cases gracefully: no devices available, CUDA errors.
6. Use standard C++ I/O (iostream) for printing, but include <cuda_runtime.h> for CUDA API.
7. The code should be self-contained, compile with nvcc, and output the required information.
8. Include basic error checking macro to simplify error handling.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <string>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error in " << __FILE__ << ":"        \
                      << __LINE__ << " - " << cudaGetErrorString(err) \
                      << std::endl;                                \
            return EXIT_FAILURE;                                    \
        }                                                           \
    } while (0)

int main() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return EXIT_FAILURE;
    }

    int bestDevice = -1;
    size_t maxGlobalMem = 0;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        // Debug: print device properties (optional)
        // std::cout << "Device " << dev << ": " << prop.name
        //           << " - " << prop.totalGlobalMem << " bytes" << std::endl;

        if (prop.totalGlobalMem > maxGlobalMem) {
            maxGlobalMem = prop.totalGlobalMem;
            bestDevice = dev;
        }
    }

    if (bestDevice >= 0) {
        cudaDeviceProp bestProp;
        CHECK_CUDA(cudaGetDeviceProperties(&bestProp, bestDevice));
        std::cout << "Device with most global memory: " << bestProp.name
                  << " (Index: " << bestDevice << ")" << std::endl;
        std::cout << "Global memory: " << maxGlobalMem << " bytes" << std::endl;
    } else {
        std::cerr << "Failed to determine device with most global memory." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```