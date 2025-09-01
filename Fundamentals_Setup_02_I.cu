```cpp
/* 
Aim of the program: For each device, query and print whether it supports running kernels concurrently.

Thinking process:
- We need to use CUDA Runtime API to get the number of CUDA devices and then query each device's properties.
- The relevant property is `concurrentKernels`, which indicates whether a device can run multiple kernels concurrently.
- Use `cudaGetDeviceCount` to obtain the total number of devices.
- Loop over each device index, call `cudaGetDeviceProperties` to get a `cudaDeviceProp` struct.
- Check the `concurrentKernels` field (an int, 0 or 1). If it's non-zero, the device supports concurrent kernels.
- Print the device index, its name, and whether it supports concurrent kernels ("Yes" or "No").
- Include error checking for CUDA API calls to ensure robustness.
- The program is written in C++ style (since CUDA C allows C++ features) but uses standard C headers.
- The output will be printed to standard output.

*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error retrieving device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error retrieving properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Supports concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("\n");
    }

    return 0;
}
```