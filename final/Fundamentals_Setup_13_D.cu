```cuda
/*
Aim: Query and print the memory clock rate of the primary GPU.

Thinking:
- The program will use the CUDA Runtime API to query properties of the primary device (device 0).
- cudaGetDeviceProperties fills a cudaDeviceProp struct, which contains a field `memoryClockRate` in kHz.
- We'll convert this value to MHz for a more readable output.
- Basic error checking will be performed for CUDA API calls.
- The program will compile as a .cu file and can be run with `nvcc` to display the memory clock rate.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            return EXIT_FAILURE;                                            \
        }                                                                    \
    } while (0)

int main(void) {
    int device = 0; // Primary GPU
    cudaDeviceProp prop;

    // Get properties of the device
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // memoryClockRate is in kHz; convert to MHz
    float memory_clock_mhz = prop.memoryClockRate / 1000.0f;

    printf("Primary GPU (device %d) memory clock rate: %.1f MHz\n",
           device, memory_clock_mhz);

    return EXIT_SUCCESS;
}
```