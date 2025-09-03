/*
Write a CUDA program to query and print the number of CUDA-enabled devices available on your system. The program should use the `cudaGetDeviceCount` API call and print a user-friendly message, such as 'Found X CUDA-capable devices.'

Thinking process:
1. The goal is to create a minimal CUDA host program that queries the system for CUDA-capable GPUs.
2. The CUDA runtime API provides `cudaGetDeviceCount(int *count)` which returns the number of devices.
3. We need to include the CUDA header `<cuda_runtime.h>` and the standard I/O header `<stdio.h>`.
4. In `main`, we declare an `int deviceCount` and call `cudaGetDeviceCount(&deviceCount)`.
5. The API returns a `cudaError_t`; we should check if it's `cudaSuccess`. If not, we should print an error message using `cudaGetErrorString(err)`.
6. If the call is successful, we print "Found X CUDA-capable devices." where X is `deviceCount`.
7. Finally, return 0 from `main`.
8. Since this is a small utility, we don't need any device kernels or memory allocations.
9. The program can be compiled with `nvcc -o device_query device_query.cu` and executed on any system with CUDA installed.
10. The code will be straightforward and self-contained.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device count: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    printf("Found %d CUDA-capable device%s.\n", deviceCount,
           (deviceCount == 1) ? "" : "s");

    return 0;
}
