```cuda
/* 
Aim of the program: Check if the device can map host memory using the `canMapHostMemory` property and print the result.

Thinking process:
1. We need to query the CUDA device properties, specifically the `canMapHostMemory` flag, which indicates whether the device supports mapping host memory into its address space.
2. To do this, we will use the CUDA Runtime API:
   - `cudaGetDeviceCount` to ensure at least one CUDA-capable device exists.
   - `cudaGetDeviceProperties` to obtain the `cudaDeviceProp` structure for a chosen device (device 0 for simplicity).
3. After retrieving the device properties, we will inspect the `canMapHostMemory` boolean field.
4. We then print the device name and the status of the `canMapHostMemory` property in a human‑readable format.
5. Error handling: If any CUDA API call fails, we output an appropriate error message and exit with a non‑zero status.
6. The program is written in C style with the CUDA Runtime API and can be compiled with `nvcc` into a .cu file.

The final code is a minimal, self‑contained CUDA program that performs the requested check and prints the result.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    // Use device 0 for demonstration
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Device 0: %s\n", prop.name);
    printf("Can map host memory: %s\n", prop.canMapHostMemory ? "Yes" : "No");

    return 0;
}
```