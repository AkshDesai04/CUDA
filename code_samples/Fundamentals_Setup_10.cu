```cpp
/*
Write a program to check and print the L2 cache size of the primary GPU in kilobytes (KB).
Thinking:
- The goal is to query the properties of the first (primary) CUDA device and display its L2 cache size in KB.
- We will use the CUDA Runtime API.
- First, check how many CUDA-capable devices are available with cudaGetDeviceCount().
- If there are no devices, we print an error message and exit.
- For the primary device (device 0), we retrieve its properties with cudaGetDeviceProperties().
- The cudaDeviceProp structure contains the field l2CacheSize which holds the size in bytes.
- Convert that value to kilobytes by dividing by 1024.
- Print the result in a readable format, e.g., "L2 cache size of primary GPU (device 0): 98304 KB".
- Handle errors from CUDA API calls by printing descriptive messages and exiting with a non-zero status.
- The program should be compilable with nvcc and run on a system with a CUDA-capable GPU.
*/

#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        std::cerr << "Error: cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA-capable device is detected." << std::endl;
        return EXIT_FAILURE;
    }

    int device = 0; // Primary device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        std::cerr << "Error: cudaGetDeviceProperties failed: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    size_t l2CacheSizeBytes = static_cast<size_t>(prop.l2CacheSize);
    size_t l2CacheSizeKB = l2CacheSizeBytes / 1024;

    std::cout << "L2 cache size of primary GPU (device " << device << "): "
              << l2CacheSizeKB << " KB" << std::endl;

    return EXIT_SUCCESS;
}
```