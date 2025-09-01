/*
Aim: Calculate and print the theoretical maximum number of concurrent warps the entire GPU can run.

Thinking:
- On NVIDIA GPUs, each streaming multiprocessor (SM) can execute a limited number of threads concurrently, specified by the property `maxThreadsPerMultiProcessor`.
- Threads are executed in groups called warps. A warp size is given by the property `warpSize` (typically 32).
- The maximum number of warps that can be active on a single SM is thus `maxThreadsPerMultiProcessor / warpSize`.
- The GPU contains multiple SMs, counted by `multiProcessorCount`.
- Therefore, the theoretical maximum number of concurrent warps the whole GPU can run is:
    `(maxThreadsPerMultiProcessor / warpSize) * multiProcessorCount`.
- The program queries the first CUDA device (device 0), retrieves these properties, performs the calculation, and prints the result.
- Error checking is performed on the CUDA API calls.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0; // Use the first CUDA device
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Compute theoretical maximum concurrent warps
    int warpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    int totalConcurrentWarps = warpsPerSM * prop.multiProcessorCount;

    printf("Device %d: %s\n", device, prop.name);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Maximum concurrent warps per SM: %d\n", warpsPerSM);
    printf("Theoretical maximum concurrent warps across the GPU: %d\n", totalConcurrentWarps);

    return 0;
}
