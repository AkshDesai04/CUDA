/*
Print the maximum number of resident blocks per multiprocessor.

The goal of this program is straightforward: query the CUDA driver for
the maximum number of blocks that can reside concurrently on a single
multiprocessor (SM) of a CUDA-capable GPU.  The CUDA runtime API
provides this information in the cudaDeviceProp structure as the field
maxBlocksPerMultiprocessor.  By inspecting that field we can learn the
theoretical limit on block residency for a device, which can be useful
when designing kernels and assessing occupancy.

Approach:
1. Include the necessary CUDA and standard headers.
2. Use cudaGetDeviceCount to determine how many GPUs are present.
3. For each device (or simply the first one), call cudaGetDeviceProperties
   to obtain a cudaDeviceProp structure.
4. Extract and print prop.maxBlocksPerMultiprocessor.
5. Perform basic error checking and report any failures.

This program compiles with nvcc and runs on any machine with a
CUDA-capable GPU.  It prints the desired value for the first device
(you can modify it to iterate over all devices if desired).
*/

#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable device found." << std::endl;
        return 1;
    }

    // Use the first device
    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Device " << device << " (" << prop.name << ") "
              << "Max resident blocks per multiprocessor: "
              << prop.maxBlocksPerMultiprocessor << std::endl;

    return 0;
}
