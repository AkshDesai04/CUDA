```cu
/*
Aim of the program: 
Check if the GPU is part of a multi-GPU board and print the result.

Thinking and design decisions:
- CUDA exposes device properties via the cudaDeviceProp structure obtained with cudaGetDeviceProperties.
- The field `multiGpuBoard` (int) indicates whether a device belongs to a multi-GPU board.
  - 0: not part of a multi-GPU board
  - 1: part of a multi-GPU board
  - 2: multi-GPU board is not supported (rare)
- We'll query all devices available on the system using cudaGetDeviceCount.
- For each device, we will:
  1. Retrieve properties.
  2. Check `multiGpuBoard`.
  3. Print a clear message indicating the device number, name, and whether it is part of a multi-GPU board.
- Error handling: A helper macro `CUDA_CHECK` will wrap CUDA API calls to abort with a message on failure.
- The program is intended to be compiled with `nvcc` and run on any system with CUDA installed.

Implementation details:
- Include <stdio.h> for printing.
- Use `cudaGetDeviceCount` and `cudaGetDeviceProperties`.
- Print `deviceName`, `multiGpuBoard` status.
- The program prints to stdout; no external files are created.

Now the actual code follows below.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro for CUDA error checking */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));       \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA-capable device(s).\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("  Clock rate: %d kHz\n", prop.clockRate);

        /* multiGpuBoard: 0 - not part of multi-GPU board
                          1 - part of multi-GPU board
                          2 - multi-GPU board support not present */
        const char *mgpStatus;
        switch (prop.multiGpuBoard) {
            case 0:
                mgpStatus = "No";
                break;
            case 1:
                mgpStatus = "Yes";
                break;
            case 2:
                mgpStatus = "Unknown (multi-GPU board support not present)";
                break;
            default:
                mgpStatus = "Invalid value";
        }
        printf("  Part of multi-GPU board: %s\n", mgpStatus);

        /* If the device is on a multi-GPU board, print its ID */
        if (prop.multiGpuBoard == 1) {
            printf("  Multi-GPU board ID: %d\n", prop.multiGpuBoardId);
        }

        printf("\n");
    }

    return EXIT_SUCCESS;
}
```