```cpp
/*
Aim of the program: 
Iterate through all devices and, for each one, print its multiprocessor count.

Thinking:
1. Use the CUDA Runtime API to query the number of CUDA-capable devices on the system.
2. For each device index from 0 to deviceCount-1, call cudaGetDeviceProperties to obtain a cudaDeviceProp struct.
3. The struct contains a field called multiProcessorCount that indicates the number of multiprocessors (SMs) on that device.
4. Print the device index, device name, and its multiprocessor count to standard output.
5. Handle any errors from CUDA calls by printing an error message and exiting.
6. Include the necessary headers: <cuda_runtime.h>, <cstdio>, <cstdlib>, <iostream>.
7. Compile this file with nvcc (e.g., nvcc -o device_info device_info.cu) and run.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0)
    {
        std::cout << "No CUDA-capable devices detected." << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "Detected " << deviceCount << " CUDA device(s)." << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            std::cerr << "cudaGetDeviceProperties failed for device " << dev << ": " 
                      << cudaGetErrorString(err) << std::endl;
            continue; // skip to next device
        }

        std::cout << "Device " << dev << ": " << prop.name
                  << " - Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
    }

    return EXIT_SUCCESS;
}
```