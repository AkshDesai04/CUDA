/*
A more direct property is `deviceOverlap`. Query and print this boolean property for the primary device.

Thinking: 
- To get the primary (current) CUDA device, call `cudaGetDevice`.
- Use `cudaGetDeviceProperties` to retrieve the properties structure.
- The `deviceOverlap` field indicates whether the device can concurrently execute a kernel and copy memory from device to host.
- Print the value as a boolean ("true" or "false") for clarity.
- Check CUDA API return codes for robust error handling.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device_id;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting current device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    struct cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("deviceOverlap for device %d: %s\n", device_id, prop.deviceOverlap ? "true" : "false");
    return 0;
}
