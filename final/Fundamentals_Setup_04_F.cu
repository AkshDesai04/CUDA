```cpp
/*
Aim: Query and print the `computeMode` of the primary GPU (e.g., Default, Exclusive, Prohibited).

Thinking:
- The compute mode of a CUDA device can be obtained from the device properties using `cudaGetDeviceProperties`. 
- The `cudaDeviceProp` struct contains a field `computeMode` which indicates the compute mode for the device. 
- We will query device 0 (the primary GPU) by calling `cudaSetDevice(0)` and then `cudaGetDeviceProperties(&prop, 0)`. 
- The `computeMode` field is an enum of type `cudaComputeMode` with values: 
    * `cudaComputeModeDefault`
    * `cudaComputeModeExclusive`
    * `cudaComputeModeProhibited`
    * `cudaComputeModeExclusiveProcess` (in newer CUDA versions)
- We will map these enum values to human‑readable strings for printing. 
- Basic error checking will be performed after each CUDA API call. 
- The program is written in C++ with the CUDA Runtime API and will compile to a .cu file.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <string>

std::string computeModeToString(cudaDeviceProp const &prop) {
    switch (prop.computeMode) {
        case cudaComputeModeDefault:
            return "Default";
        case cudaComputeModeExclusive:
            return "Exclusive";
        case cudaComputeModeProhibited:
            return "Prohibited";
#ifdef cudaComputeModeExclusiveProcess
        case cudaComputeModeExclusiveProcess:
            return "ExclusiveProcess";
#endif
        default:
            return "Unknown";
    }
}

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return -1;
    }

    int device = 0; // primary GPU
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set device " << device << ": " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties for device " << device << ": " 
                  << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Device " << device << " (" << prop.name << "): Compute Mode = "
              << computeModeToString(prop) << std::endl;

    return 0;
}
```