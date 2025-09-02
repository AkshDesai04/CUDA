```cpp
/*
Trigger `cudaErrorNoDevice` by trying to run a CUDA program on a system with no NVIDIA driver or no CUDA-capable GPU.

Thinking:
The goal is to generate the CUDA error code cudaErrorNoDevice, which indicates that no CUDA-capable device is available or that the driver is missing. The simplest way to trigger this is to invoke a CUDA runtime API that requires at least one device, such as cudaGetDeviceCount or cudaSetDevice. When these functions are called on a system without a GPU or without the proper driver, they return cudaErrorNoDevice. The program below calls cudaGetDeviceCount, checks for errors, and prints a helpful message. If no device is found, the error string from cudaGetErrorString will indicate the issue. This code is deliberately minimal and self-contained so it can be compiled with nvcc and run on any machine; it will fail with the desired error when no CUDA hardware is present.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);
    return 0;
}
```