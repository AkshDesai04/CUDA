```c
/*
Write a function `get_gpu_memory_mb(int device_id)` that returns the total global memory in MB for the given device.
This comment explains the program's purpose exactly as requested, followed by a detailed reasoning of how to implement it.

Thinking:
1. The goal is to obtain the total global memory of a CUDA-enabled GPU specified by `device_id`.
2. CUDA provides `cudaGetDeviceProperties` which fills a `cudaDeviceProp` struct containing a field `totalGlobalMem` with the memory in bytes.
3. No kernel launches are required; we only query device properties, which is a host operation.
4. The function should handle errors: if `cudaGetDeviceProperties` returns a non-`cudaSuccess` code, we return 0 or some sentinel (here 0) and optionally print an error.
5. Convert bytes to megabytes by dividing by `(1024 * 1024)` and cast to `int`.
6. For safety, we might set the device using `cudaSetDevice(device_id)` first, though `cudaGetDeviceProperties` can query any device without setting it. Setting the device also ensures the context is created for that device.
7. The function signature is `int get_gpu_memory_mb(int device_id)` returning MB.
8. A small `main` is provided to demonstrate usage: it prints the memory of device 0 if available.
9. All necessary headers and error-checking macros are included.

The resulting .cu file can be compiled with nvcc and run on a system with CUDA installed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Utility macro for CUDA error checking
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            return 0; /* Return 0 as error indicator */                    \
        }                                                                   \
    } while (0)

/**
 * @brief Retrieves the total global memory of the specified CUDA device in megabytes.
 *
 * @param device_id The ID of the CUDA device (0-based indexing).
 * @return The total global memory in MB. Returns 0 on error or if the device ID is invalid.
 */
int get_gpu_memory_mb(int device_id)
{
    // First, check if the device ID is valid by trying to get its properties.
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    // Convert bytes to megabytes
    size_t mem_bytes = prop.totalGlobalMem;
    int mem_mb = (int)(mem_bytes / (1024 * 1024));

    return mem_mb;
}

int main()
{
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < device_count; ++dev) {
        int mem_mb = get_gpu_memory_mb(dev);
        if (mem_mb > 0) {
            printf("Device %d has %d MB of total global memory.\n", dev, mem_mb);
        } else {
            printf("Failed to get memory for device %d.\n", dev);
        }
    }

    return 0;
}
```