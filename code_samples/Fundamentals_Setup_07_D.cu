/*
Query and check if the device supports Page-locked Memory Mapped On The GPU (`pageableMemoryAccess`).

The goal of this program is to determine whether the currently selected CUDA device
supports the capability to map page-locked (pinned) memory directly into the device
address space. This is an important feature for performance-critical applications
that wish to avoid costly explicit data copies and instead perform zero-copy
operations. The capability is exposed via the `cudaDeviceProp::pageableMemoryAccess`
field returned by `cudaGetDeviceProperties`. A value of 1 indicates that the
device supports page-locked memory mapping, whereas a value of 0 indicates it
does not.

The program performs the following steps:
1. Uses `cudaGetDeviceCount` to determine how many CUDA-capable devices are
   available.
2. Iterates over each device, querying its properties with `cudaGetDeviceProperties`.
3. Extracts the `pageableMemoryAccess` flag from the `cudaDeviceProp` structure.
4. Prints the device name, compute capability, total global memory, and whether
   pageable memory mapping is supported.
5. Reports any CUDA API errors in a clear manner.

The code is written in standard C++ using the CUDA Runtime API, making it
portable across CUDA-capable GPUs. No external dependencies are required
besides the CUDA toolkit.

This program provides a quick diagnostic tool for developers who need to know
whether they can rely on page-locked memory mapping on their target device(s).
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>

// Macro to check CUDA API call results
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Detected " << deviceCount << " CUDA-capable device(s)." << std::endl << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << static_cast<double>(prop.totalGlobalMem) / (1 << 20)
                  << " MB" << std::endl;
        std::cout << "  Pageable Memory Access: "
                  << (prop.pageableMemoryAccess ? "Supported" : "Not Supported") << std::endl;

        // Additional info: Whether the device can map host memory directly
        std::cout << "  Host Memory Mapping: "
                  << (prop.canMapHostMemory ? "Supported" : "Not Supported") << std::endl;

        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
