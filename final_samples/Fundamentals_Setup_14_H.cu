```c
/* 
Aim of the program:
Check if the device supports host-mapped pageable memory access (`pageableMemoryAccess`).

Thinking:
To determine whether the CUDA-capable device allows host-mapped pageable memory access, we need to inspect the device's properties exposed by the CUDA Runtime API. The `cudaDeviceProp` structure contains a boolean flag `pageableMemoryAccess` that indicates this capability. The steps are:
1. Retrieve the current device ID with `cudaGetDevice()`.
2. Allocate a `cudaDeviceProp` structure and fill it using `cudaGetDeviceProperties()`.
3. Read the `pageableMemoryAccess` field.
4. Print a friendly message indicating whether the feature is supported.
This program compiles with `nvcc` and prints the result to the console.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d: %s\n", device, prop.name);
    if (prop.pageableMemoryAccess) {
        printf("Host-mapped pageable memory access (pageableMemoryAccess) is supported.\n");
    } else {
        printf("Host-mapped pageable memory access (pageableMemoryAccess) is NOT supported.\n");
    }

    return 0;
}
```