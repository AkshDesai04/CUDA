/*
Query and print the computeMode of the primary GPU (e.g., Default, Exclusive, Prohibited).

Thought process:
- The goal is to determine the compute mode set on the device that the CUDA runtime
  considers primary (the device that the program is currently running on).  
- In CUDA, each device has a compute mode that can be queried via `cudaGetDeviceProperties`,
  which returns a `cudaDeviceProp` structure containing a `computeMode` field.
- The `cudaComputeMode` enum includes values: `cudaComputeModeDefault`, `cudaComputeModeExclusive`,
  `cudaComputeModeProhibited`, and `cudaComputeModeExclusiveProcess`.  
- The program will:
  1. Retrieve the current device ID using `cudaGetDevice`.
  2. Query the device properties with `cudaGetDeviceProperties`.
  3. Map the enum value to a readable string.
  4. Print the result along with the device name and ID.
- Error handling will be added to report any failures from CUDA API calls.
- The code is written in standard C++ with CUDA runtime API and will be compiled as a
  `.cu` file. No additional headers beyond `<cuda_runtime.h>` and `<iostream>` are required.

*/

#include <cuda_runtime.h>
#include <iostream>
#include <string>

std::string computeModeToString(cudaComputeMode mode) {
    switch (mode) {
        case cudaComputeModeDefault:          return "Default";
        case cudaComputeModeExclusive:        return "Exclusive";
        case cudaComputeModeProhibited:       return "Prohibited";
        case cudaComputeModeExclusiveProcess: return "ExclusiveProcess";
        default:                              return "Unknown";
    }
}

int main() {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get current device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Compute mode of device " << device << " (" << prop.name << "): "
              << computeModeToString(prop.computeMode) << std::endl;

    return 0;
}
