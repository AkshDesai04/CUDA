/*
Aim: Check if the GPU is part of a multi-GPU board and print the result.

Thought Process:
- The CUDA runtime provides device properties through cudaGetDeviceProperties().
- The cudaDeviceProp structure includes a field named `multiGPUBoard` (int), which is set to 1 if the GPU is part of a multi-GPU board, otherwise 0.
- We'll query the total number of devices with cudaGetDeviceCount().
- For each device, retrieve its properties and examine the `multiGPUBoard` flag.
- Print a clear message indicating whether each GPU is on a multi-GPU board.
- Include basic error handling to report CUDA API failures.

The program is a self‑contained .cu file that compiles with nvcc and can be executed on a system with CUDA installed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            return EXIT_FAILURE;                                            \
        }                                                                    \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Detected %d CUDA-capable device(s).\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Multi-GPU board: %s\n\n",
               prop.multiGPUBoard ? "Yes" : "No");
    }

    return EXIT_SUCCESS;
}
