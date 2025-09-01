/*
Check if the GPU is part of a multi-GPU board and print the result.

Thinking:

1. The CUDA Runtime API provides a structure cudaDeviceProp that contains many fields describing a GPU device.
2. One of those fields is `multiGpuBoard` (an int). According to the CUDA documentation, if this value is non-zero,
   the device is part of a multi‑GPU board (e.g., an NVLink or SLI configuration). If it is zero, the device is a standalone GPU.
3. The program should:
   a. Query the number of CUDA-capable devices with `cudaGetDeviceCount`.
   b. For each device, obtain its properties with `cudaGetDeviceProperties`.
   c. Inspect the `multiGpuBoard` field and print whether the device is on a multi-GPU board.
4. For clarity, the program will output the device index, name, and the status (Yes/No).
5. Error handling: check return values of CUDA calls and print any errors encountered.
6. The code will be written in C/C++ (CUDA) and compiled with `nvcc`. It will be a single .cu file that can be built
   and run to see the result.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    printf("Detected %d CUDA-capable device(s).\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaGetDeviceProperties for device %d failed: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Multi-GPU board: %s\n\n", prop.multiGpuBoard ? "Yes" : "No");
    }

    return 0;
}
