```cuda
/* 
Aim of the program: Check if the GPU is part of a multi-GPU board and print the result.

Thinking:
- CUDA provides a device property `cudaDeviceProp::multiGpuBoard` that indicates whether
  the device is part of a multi‑GPU board (SLI or NVLink). 
- We'll query the number of CUDA devices with `cudaGetDeviceCount`.
- For simplicity, we'll check the first device (device 0). In a multi‑GPU system this
  property will be set for each device that is part of a multi‑GPU board.
- We handle error cases where the device count is zero or CUDA calls fail.
- The program outputs a clear message indicating the status of the device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    // Check the first device (index 0)
    int dev = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Device %d: %s\n", dev, prop.name);
    if (prop.multiGpuBoard) {
        printf("This GPU is part of a multi-GPU board.\n");
    } else {
        printf("This GPU is not part of a multi-GPU board.\n");
    }

    return EXIT_SUCCESS;
}
```