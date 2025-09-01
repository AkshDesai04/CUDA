/*
Aim: Write a program that finds the device with the highest compute capability and prints its name and index.

Thinking:
- We need to use the CUDA Runtime API to query available devices.
- First call cudaGetDeviceCount to find how many GPUs are present.
- Then iterate over each device index, calling cudaGetDeviceProperties to obtain a cudaDeviceProp structure.
- Compute a comparable value for each device's compute capability. A simple way is to combine the major and minor versions into a single integer: major * 100 + minor. This ensures that e.g. 6.1 > 6.0 > 5.2, etc.
- Keep track of the device index with the highest computed value.
- After examining all devices, print the index and name of the selected device using printf.
- Include basic error checking after each CUDA call to handle situations where the runtime API fails.
- The program is self‑contained, uses only standard headers plus cuda_runtime.h, and can be compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple macro for error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                   \
        }                                                           \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    int bestDeviceIdx = -1;
    int bestComputeCap = -1; // Represented as major*100 + minor

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        int computeCap = prop.major * 100 + prop.minor;
        if (computeCap > bestComputeCap) {
            bestComputeCap = computeCap;
            bestDeviceIdx = dev;
        }
    }

    if (bestDeviceIdx >= 0) {
        cudaDeviceProp bestProp;
        CUDA_CHECK(cudaGetDeviceProperties(&bestProp, bestDeviceIdx));
        printf("Highest compute capability device:\n");
        printf("  Index: %d\n", bestDeviceIdx);
        printf("  Name:  %s\n", bestProp.name);
        printf("  Compute Capability: %d.%d\n",
               bestProp.major, bestProp.minor);
    }

    return EXIT_SUCCESS;
}
