/*
Check if the GPU supports `unifiedAddressing` and print a confirmation message.
The goal of this program is straightforward: we want to determine whether the
current CUDA-capable device on the machine supports unified addressing.
Unified addressing means that the device and host share the same memory
address space, which allows kernels to reference host pointers directly
without explicit memory copies.  This is particularly useful for hybrid
CPU‑GPU applications.

Approach:
1. Use cudaGetDevice to get the current device index.
2. Query the device properties with cudaGetDeviceProperties.
3. Inspect the unifiedAddressing flag in the cudaDeviceProp struct.
4. Print a clear message indicating support or lack thereof, along with
   the device name for clarity.
5. Handle any errors from CUDA runtime calls gracefully.

This program is minimal and portable across CUDA-enabled systems. It
requires a compiler that supports CUDA, such as nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        printf("Error getting current device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        printf("Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (prop.unifiedAddressing) {
        printf("Unified addressing is supported on GPU %d (%s).\n",
               device, prop.name);
    } else {
        printf("Unified addressing is NOT supported on GPU %d (%s).\n",
               device, prop.name);
    }

    return 0;
}
