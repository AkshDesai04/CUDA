```cpp
/*
Aim of the program: Calculate and print the theoretical maximum number of concurrent warps the entire GPU can run.

Thinking:
The theoretical maximum number of concurrent warps is determined by the GPU architecture.
For each Streaming Multiprocessor (SM), CUDA can run a certain maximum number of threads
simultaneously, given by the device property maxThreadsPerMultiProcessor.
Since a warp consists of warpSize threads (typically 32), the maximum number of concurrent
warps per SM is simply maxThreadsPerMultiProcessor / warpSize.
The entire GPU contains multiProcessorCount SMs, so the total theoretical maximum number of
concurrent warps is:
    total_warps = multiProcessorCount * (maxThreadsPerMultiProcessor / warpSize)

The program will:
1. Query the number of CUDA devices.
2. For each device, retrieve its properties.
3. Compute total_warps using the formula above.
4. Print the result.

We will use the CUDA Runtime API functions:
- cudaGetDeviceCount to get the number of devices.
- cudaGetDeviceProperties to fetch the device properties.

The program will output a single integer per device, indicating the theoretical maximum
concurrent warps for that GPU.

Error handling is minimal; we check the return status of CUDA API calls and print an error
message if something goes wrong. The program is self‑contained in a single .cu file and
requires a CUDA-capable compiler (nvcc) to build.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }

        int warpSize = prop.warpSize;  // Typically 32
        int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
        int maxWarpsPerSM = maxThreadsPerSM / warpSize;

        int totalWarps = prop.multiProcessorCount * maxWarpsPerSM;

        printf("Device %d (%s): theoretical maximum concurrent warps = %d\n",
               dev, prop.name, totalWarps);
    }

    return 0;
}
```