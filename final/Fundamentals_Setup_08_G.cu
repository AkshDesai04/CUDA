/*
Aim of the program: Check if the device is a multi-GPU board (`isMultiGpuBoard`).

The goal is to query the CUDA runtime for each available GPU device, retrieve its
properties via `cudaGetDeviceProperties`, and inspect the `multiGpuBoard` flag
which is defined in `cudaDeviceProp`. This flag indicates whether the device
is part of a multi-GPU board (e.g., an NVIDIA NVLink-connected multi‑GPU
board). The program will iterate over all detected devices, print their
basic information (name, compute capability, total memory) and state whether
each is a multi-GPU board.

Approach:
1. Include `<cuda_runtime.h>` and `<stdio.h>`.
2. Define a small helper macro `CHECK_CUDA` for error checking.
3. In `main`, query the number of devices with `cudaGetDeviceCount`.
4. Loop over each device:
   - Retrieve its properties with `cudaGetDeviceProperties`.
   - Print the device index, name, compute capability, total memory.
   - Print `multiGpuBoard: true` if the flag is non‑zero, otherwise `false`.
5. Exit with 0 on success.

This program is self‑contained and can be compiled with `nvcc` as:
`nvcc -o check_mgpu check_mgpu.cu`

*/

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            return EXIT_FAILURE;                                          \
        }                                                                 \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA device(s).\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multi-GPU board: %s\n\n",
               prop.multiGpuBoard ? "true" : "false");
    }

    return EXIT_SUCCESS;
}
