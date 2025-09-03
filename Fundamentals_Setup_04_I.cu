/*
Check if the GPU supports `unifiedAddressing` and print a confirmation message.

My thinking: The CUDA runtime provides device properties via cudaGetDeviceProperties, which includes a field "unifiedAddressing" that indicates whether the device supports unified virtual addressing. The program will query the first device (device 0) or allow selecting device. Then check the property. Print a message accordingly. We'll handle errors via cudaGetLastError or check return codes. This will compile as .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found.\n");
        return -1;
    }

    // Use device 0 for simplicity
    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (prop.unifiedAddressing) {
        printf("Unified addressing supported on device %d (%s).\n", device, prop.name);
    } else {
        printf("Unified addressing NOT supported on device %d (%s).\n", device, prop.name);
    }

    return 0;
}
