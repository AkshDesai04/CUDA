```cpp
/*
Aim of the program: Check if the device can map host memory using the `canMapHostMemory` property and print the result.

Thought Process:
- The goal is to query the CUDA device properties and inspect the `canMapHostMemory` field of the `cudaDeviceProp` struct.
- We need to include the CUDA runtime header `cuda_runtime.h` and the standard I/O header.
- Use `cudaGetDeviceCount` to determine how many devices are available.
- For simplicity, we query the first device (device 0). In a more robust program we could iterate over all devices.
- Call `cudaGetDeviceProperties` to fill a `cudaDeviceProp` struct.
- Check the `canMapHostMemory` flag; if it's non-zero, the device supports mapping host memory, otherwise it does not.
- Print a clear message indicating whether the device can map host memory.
- Include basic error checking for CUDA API calls to handle unexpected failures.
- The program will be compiled with `nvcc` and run on a system with CUDA installed.

This simple program demonstrates how to inspect a device capability that is relevant for CUDA memory management, specifically host memory mapping. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable device detected.\n");
        return 0;
    }

    // Use the first device (device 0)
    int deviceId = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, deviceId);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", deviceId, cudaGetErrorString(err));
        return 1;
    }

    // Check the canMapHostMemory flag
    if (prop.canMapHostMemory) {
        printf("Device %d (%s) can map host memory: Yes\n", deviceId, prop.name);
    } else {
        printf("Device %d (%s) can map host memory: No\n", deviceId, prop.name);
    }

    return 0;
}
```