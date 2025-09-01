```c
/* 
Aim: Write a program to check if the primary GPU can overlap data copying with kernel execution. 
This capability is crucial for performance and is indicated by the asyncEngineCount property (a value > 0).

Thinking:
1. Use CUDA Runtime API to query the current device.
2. Retrieve device properties via cudaGetDeviceProperties.
3. Inspect the asyncEngineCount field; a value greater than zero indicates support for overlapping
   memory copy operations with kernel execution.
4. Report the result to the user.
5. Handle any CUDA errors with simple error checking.
6. Keep the code minimal and portable, compiling with nvcc as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0; // Primary device
    cudaDeviceProp prop;
    cudaError_t err;

    // Get current device (default 0)
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current CUDA device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Retrieve device properties
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get properties for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    // Check asyncEngineCount
    if (prop.asyncEngineCount > 0) {
        printf("Primary GPU (device %d) supports overlapping data copying with kernel execution (asyncEngineCount = %d).\n",
               device, prop.asyncEngineCount);
    } else {
        printf("Primary GPU (device %d) does NOT support overlapping data copying with kernel execution (asyncEngineCount = %d).\n",
               device, prop.asyncEngineCount);
    }

    return 0;
}
```