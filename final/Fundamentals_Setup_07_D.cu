```c
/*
Aim of the program: Query and check if the device supports Page-locked Memory Mapped On The GPU (pageableMemoryAccess).

Thought process:
1. The CUDA runtime API provides an attribute called `cudaDevAttrPageableMemoryAccess` that indicates whether a device can map pageable host memory onto the GPU without explicit registration. If this attribute is set to 1 for a device, it supports pageable memory access; otherwise, it does not.
2. To query this attribute, we will:
   - Determine the number of CUDA devices available.
   - Select a device (device 0 for simplicity, but this can be adapted).
   - Call `cudaDeviceGetAttribute` with the attribute `cudaDevAttrPageableMemoryAccess` and the chosen device index.
3. We will handle any CUDA errors that might arise from device queries or attribute retrieval.
4. Finally, we will print a clear message indicating whether the selected device supports pageable memory access.
5. The program is written in standard C with CUDA extensions and compiled as a `.cu` file using `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    int device = 0; // Use device 0 for this query
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int pageableSupport = 0;
    err = cudaDeviceGetAttribute(&pageableSupport, cudaDevAttrPageableMemoryAccess, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaDeviceGetAttribute failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d: %s\n", device, prop.name);
    if (pageableSupport) {
        printf("The device supports pageable memory access (Page-locked Memory Mapped On The GPU).\n");
    } else {
        printf("The device does NOT support pageable memory access.\n");
    }

    return 0;
}
```