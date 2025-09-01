/*
Query and print the maximum dimensions (width and height) of a 2D texture for device 0.

Thought process:
- The goal is to query the hardware capabilities of the first CUDA device (device 0) and report the maximum size that can be used for a 2D texture.
- Use the CUDA runtime API, specifically cudaGetDeviceProperties, which fills a cudaDeviceProp structure with all relevant properties of a device.
- Device 0 is selected by calling cudaSetDevice(0). This is not strictly required for cudaGetDeviceProperties, but it ensures that the device context is set correctly.
- The cudaDeviceProp structure contains a field `maxTexture2D` of type dim3, which holds the maximum width, height, and depth for 2D textures. We read the `width` and `height` components.
- Provide error checking after each CUDA call so that any issues are reported to stderr.
- Output the device name, width, and height using printf to stdout.
- The program is a minimal, self‑contained CUDA host program written in C/C++ that can be compiled with `nvcc` into an executable.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    const int device = 0;

    // Set the active device to device 0
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    // Retrieve device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get properties for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    // Output the maximum 2D texture dimensions
    printf("Device %d: %s\n", device, prop.name);
    printf("Maximum 2D texture width:  %d\n", prop.maxTexture2D.width);
    printf("Maximum 2D texture height: %d\n", prop.maxTexture2D.height);

    return 0;
}
