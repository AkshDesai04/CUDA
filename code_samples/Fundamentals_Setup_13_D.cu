```cu
/*
Aim: Query and print the memory clock rate of the primary GPU.

Thinking process:

1. The CUDA Runtime API provides a function cudaGetDeviceProperties that fills a cudaDeviceProp structure with device information.
2. The memory clock rate is stored in the memoryClockRate field of this structure, expressed in kHz.
3. To get the "primary GPU", we simply query device 0 (the first GPU) which is typically the primary one.
4. We need to handle possible errors from the CUDA API calls. A simple error checking macro will be defined for readability.
5. The output should be printed in a human-readable format, converting kHz to MHz for convenience.
6. The program will be a minimal self-contained CUDA C file that can be compiled with nvcc and executed on a machine with a CUDA-capable GPU.

Implementation steps:
- Include <cuda_runtime.h> and <stdio.h>.
- Define a macro CHECK_CUDA(call) that checks the return value of a CUDA API call and exits on error.
- In main():
    * Call CHECK_CUDA(cudaGetDeviceCount(&deviceCount)) to ensure at least one GPU exists.
    * Use device 0; call CHECK_CUDA(cudaSetDevice(0)).
    * Declare a cudaDeviceProp struct and call CHECK_CUDA(cudaGetDeviceProperties(&prop, 0)).
    * Print the memory clock rate: prop.memoryClockRate (kHz). Convert to MHz by dividing by 1000.0.
- Return 0.

This program is straightforward and demonstrates basic CUDA device introspection.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                 \
    do {                                                 \
        cudaError_t err = (call);                        \
        if (err != cudaSuccess) {                        \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                          \
        }                                                \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    int deviceId = 0; // Primary GPU
    CHECK_CUDA(cudaSetDevice(deviceId));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));

    // memoryClockRate is in kHz; convert to MHz
    double memoryClockMHz = prop.memoryClockRate / 1000.0;
    printf("Device %d (%s) memory clock rate: %.2f MHz\n",
           deviceId, prop.name, memoryClockMHz);

    return EXIT_SUCCESS;
}
```