/*
For each device, print its L2 cache size in megabytes (MB).

Thinking:
1. Use the CUDA Runtime API to query the number of CUDA-capable devices via `cudaGetDeviceCount`.
2. Loop over each device index from 0 to count-1.
3. For each device, obtain its properties using `cudaGetDeviceProperties`. The `cudaDeviceProp` structure contains a field `l2CacheSize` that holds the size of the L2 cache in bytes (if the device supports it).
4. Convert the L2 cache size from bytes to megabytes by dividing by (1024*1024).
5. Print the device index and its L2 cache size in a human-readable format.
6. Add basic error handling for the CUDA API calls.
7. The program is self‑contained and can be compiled with `nvcc`. It simply outputs the L2 cache sizes for all detected devices when executed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Unable to get device count: %s\n", cudaGetErrorString(err));
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
            fprintf(stderr, "Error: Unable to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        size_t l2CacheBytes = prop.l2CacheSize;
        double l2CacheMB = (double)l2CacheBytes / (1024.0 * 1024.0);

        printf("Device %d: L2 cache size = %.2f MB\n", dev, l2CacheMB);
    }

    return 0;
}
