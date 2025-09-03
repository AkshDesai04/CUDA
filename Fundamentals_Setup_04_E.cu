/*
Calculate and print the theoretical maximum number of concurrent warps the entire GPU can run.

Thinking:
We need to query CUDA device properties to determine how many warps can run concurrently on each streaming multiprocessor (SM). The device property `maxThreadsPerMultiProcessor` gives the maximum number of threads that can reside on an SM at once. Since a warp consists of `warpSize` threads (usually 32), the theoretical maximum number of active warps per SM is `maxThreadsPerMultiProcessor / warpSize`. The total theoretical maximum across the entire GPU is this per-SM value multiplied by the number of SMs (`multiProcessorCount`). We'll use the CUDA Runtime API to get device properties (`cudaGetDeviceProperties`) and compute this value for each device, printing the result. We will also include basic error checking for CUDA calls.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable device found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        int warpSize = prop.warpSize; // usually 32
        int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
        int maxWarpsPerSM = maxThreadsPerSM / warpSize;

        // Total theoretical concurrent warps across all SMs
        int totalConcurrentWarps = maxWarpsPerSM * prop.multiProcessorCount;

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Multi-Processor Count (SMs): %d\n", prop.multiProcessorCount);
        printf("  Warp Size: %d threads\n", warpSize);
        printf("  Max Threads per SM: %d\n", maxThreadsPerSM);
        printf("  Max Active Warps per SM (theoretical): %d\n", maxWarpsPerSM);
        printf("  Theoretical maximum concurrent warps across GPU: %d\n\n", totalConcurrentWarps);
    }

    return EXIT_SUCCESS;
}
