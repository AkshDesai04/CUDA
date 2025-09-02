```cuda
/*
Aim of the program: Write a function `get_gpu_memory_mb(int device_id)` that returns the total global memory in MB for the given device.

Thought process:
1. The CUDA Runtime API provides device properties via `cudaGetDeviceProperties`. One of the fields in `cudaDeviceProp` is `totalGlobalMem`, which gives the total memory in bytes.
2. To convert bytes to megabytes, divide by 1024*1024.
3. The function should handle errors: if `cudaGetDeviceProperties` fails, print an error message and return -1 to indicate failure.
4. The function will return an `int` representing megabytes. It could also return a `long long` or `size_t`, but `int` is sufficient for typical GPU memory sizes.
5. Include necessary headers (`cuda_runtime.h` and `stdio.h`).
6. Provide a small `main` function to demonstrate usage: it will query the number of devices, pick device 0, and print its memory size in MB.
7. Ensure the code compiles as a CUDA C source file (`.cu`), so we use `__global__`/`__device__` only if needed; here everything is host code.
8. Add error checks and use `cudaGetErrorString` to print readable error messages.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Function that returns the total global memory of a GPU device in megabytes.
// Returns -1 on error.
int get_gpu_memory_mb(int device_id)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n",
                device_id, cudaGetErrorString(err));
        return -1;
    }

    size_t total_bytes = prop.totalGlobalMem;
    int total_mb = (int)(total_bytes / (1024 * 1024));
    return total_mb;
}

// Simple main to demonstrate the usage of get_gpu_memory_mb.
int main(void)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (device_count == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Number of CUDA devices: %d\n", device_count);

    for (int dev = 0; dev < device_count; ++dev) {
        int mem_mb = get_gpu_memory_mb(dev);
        if (mem_mb >= 0) {
            printf("Device %d: %d MB of total global memory\n", dev, mem_mb);
        } else {
            printf("Device %d: Unable to query memory\n", dev);
        }
    }

    return 0;
}
```