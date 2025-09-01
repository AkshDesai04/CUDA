/*
Aim: Query and print the total number of registers available per thread block on the primary device.
Thinking: 
- Use the CUDA runtime API to query device properties.
- The structure `cudaDeviceProp` contains the field `regsPerBlock`, which represents the maximum number of 32‑bit registers that can be allocated per thread block on a device.
- We will first retrieve the current device with `cudaGetDevice()`, then obtain its properties via `cudaGetDeviceProperties()`.
- Finally, we print the value of `regsPerBlock`.
- Basic error handling is added to catch any CUDA runtime errors during the queries.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting current device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Total number of registers available per thread block on device %d (%s): %d\n",
           device, prop.name, prop.regsPerBlock);

    return 0;
}
