/*
For each device, print its memory bus width in bits.
Thought process:
- Use CUDA runtime API to get the number of devices (cudaGetDeviceCount).
- For each device index, retrieve device properties with cudaGetDeviceProperties.
- The memory bus width is stored in the deviceProperties.memoryBusWidth field (in bits).
- Print the device number, name, and memory bus width.
- Include error checking for CUDA API calls using a macro to keep the code concise.
- Output is directed to stdout.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            return EXIT_FAILURE;                                    \
        }                                                           \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Detected %d CUDA device(s).\n", deviceCount);
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
    }

    return EXIT_SUCCESS;
}
