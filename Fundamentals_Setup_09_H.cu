/*
Aim: Check if the device supports local L1 cache (`localL1CacheSupported`).

Thinking:
- The CUDA Runtime API provides device attributes that can be queried using `cudaDeviceGetAttribute`.
- The attribute for local L1 cache support is `cudaDevAttrLocalL1CacheSupported`. This attribute was added in recent CUDA versions.
- To determine support, we first obtain the number of CUDA-capable devices with `cudaGetDeviceCount`.
- For each device, we query the attribute. If the call succeeds, we interpret the returned integer (non-zero means supported).
- If the attribute is not supported by the runtime (e.g., older CUDA versions), `cudaDeviceGetAttribute` will return `cudaErrorNotSupported`. In that case, we treat it as not supported and inform the user accordingly.
- The program prints the result for each device in a readable format.

The code uses standard CUDA headers and C++ i/o for simplicity. It gracefully handles errors and reports them.

*/

#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0)
    {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "Found " << deviceCount << " CUDA-capable device(s)." << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        int supported = 0;
        err = cudaDeviceGetAttribute(&supported, cudaDevAttrLocalL1CacheSupported, dev);
        std::cout << "Device " << dev << ": ";
        if (err == cudaSuccess)
        {
            std::cout << "localL1CacheSupported = " << (supported ? "true" : "false");
        }
        else if (err == cudaErrorNotSupported)
        {
            std::cout << "Attribute not supported by this CUDA runtime version.";
        }
        else
        {
            std::cout << "Error querying attribute: " << cudaGetErrorString(err);
        }
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
