/*
Aim: Query and print the GPU's core clock rate in GHz.

Thinking:
- The CUDA runtime API provides the device properties through cudaDeviceProp.
- The member `clockRate` in `cudaDeviceProp` is expressed in kHz.
- To convert this to GHz, multiply by 1e-6 (i.e., divide by 1,000,000).
- We'll query the first CUDA-capable device (device 0) for simplicity.
- Error handling will be included for `cudaGetDeviceCount` and `cudaGetDeviceProperties`.
- The result will be printed using `printf`.
- The code uses standard C headers and the CUDA runtime header.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 1;
    }

    int device = 0; // Select the first device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        printf("Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    double coreClockGHz = prop.clockRate / 1e6; // Convert kHz to GHz
    printf("GPU Device %d (%s) core clock rate: %.3f GHz\n", device, prop.name, coreClockGHz);

    return 0;
}
