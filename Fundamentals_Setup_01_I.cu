/*
 * Aim: Query and print the PCI Bus ID and PCI Device ID for the primary GPU.
 *
 * Thinking:
 * - We want to query the primary GPU, which is device 0 in CUDA.
 * - Use cudaGetDeviceProperties to obtain a cudaDeviceProp struct.
 * - The struct contains `pciBusID` and `pciDeviceID` fields (note that
 *   the fields are named `pciBusID` and `pciDeviceID` in recent CUDA
 *   versions; older versions used `busId` and `deviceId`).
 * - We'll perform error checking after each CUDA call.
 * - Print the bus and device IDs in a readable format.
 * - The program is a simple console application written in C++ (compatible
 *   with CUDA C++). It can be compiled with `nvcc` and run on a machine
 *   with an NVIDIA GPU.
 */

#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in file '" << __FILE__                 \
                      << "' in line " << __LINE__ << ": "                   \
                      << cudaGetErrorString(err) << " (" << err << ")"      \
                      << std::endl;                                         \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return EXIT_FAILURE;
    }

    // Query properties for the primary GPU (device 0)
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    // Print PCI Bus ID and PCI Device ID
    std::cout << "Primary GPU PCI Bus ID: "   << static_cast<int>(deviceProp.pciBusID)   << std::endl;
    std::cout << "Primary GPU PCI Device ID: " << static_cast<int>(deviceProp.pciDeviceID) << std::endl;

    return EXIT_SUCCESS;
}
