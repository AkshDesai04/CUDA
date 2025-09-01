/*
Aim of the program: Query and print the `asyncEngineCount` property, which indicates the number of engines available for overlapping data copies and kernel execution.

Thinking:
1. The asyncEngineCount is a field in the cudaDeviceProp struct, which is retrieved via cudaGetDeviceProperties().
2. To obtain it for all available GPUs, we first call cudaGetDeviceCount() to determine how many devices are present.
3. Iterate over each device index, call cudaGetDeviceProperties() for that index, and print the asyncEngineCount value.
4. Add basic error checking after each CUDA API call to handle potential failures.
5. Use std::cout to output the results in a readable format.
6. The program is written in CUDA C and compiles as a .cu file, making use of the CUDA Runtime API.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error retrieving device count: " << cudaGetErrorString(err) << std::endl;
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
            std::cerr << "Error retrieving properties for device " << dev << ": " << cudaGetErrorString(err) << std::endl;
            continue;
        }
        std::cout << "Device " << dev << ": asyncEngineCount = " << prop.asyncEngineCount << std::endl;
    }

    return 0;
}
