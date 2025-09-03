/*
Write a function `get_gpu_memory_mb(int device_id)` that returns the total global memory in MB for the given device.

Thinking:

1. The CUDA runtime provides `cudaGetDeviceProperties` which fills a `cudaDeviceProp` structure with details about a device, including `totalGlobalMem` which is in bytes.
2. To implement `get_gpu_memory_mb`, we will:
   a. Validate the device_id using `cudaGetDeviceProperties`. If the device does not exist or an error occurs, we return -1 or 0 to indicate failure.
   b. Retrieve the `totalGlobalMem` field.
   c. Convert bytes to megabytes: divide by (1024 * 1024).
   d. Return the integer megabyte value.
3. For robustness, we will check the return status of `cudaGetDeviceProperties`. If it fails, we print an error message to stderr and return -1.
4. Optionally, a `main` function can be added to demonstrate usage: it will query the memory for device 0 and print it.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int get_gpu_memory_mb(int device_id)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: cudaGetDeviceProperties failed for device %d: %s\n",
                device_id, cudaGetErrorString(err));
        return -1;
    }

    // Convert bytes to megabytes
    long long totalBytes = static_cast<long long>(prop.totalGlobalMem);
    int totalMB = static_cast<int>(totalBytes / (1024 * 1024));

    return totalMB;
}

#ifdef DEMO_MAIN
int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        int memMB = get_gpu_memory_mb(dev);
        if (memMB >= 0)
            printf("Device %d has %d MB of global memory.\n", dev, memMB);
        else
            printf("Failed to get memory for device %d.\n", dev);
    }

    return 0;
}
#endif
