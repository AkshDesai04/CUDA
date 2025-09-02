```c
/*
Aim of the program: For each device, print its L2 cache size in megabytes (MB).

Thinking process:
1. The goal is to query all CUDA-capable devices present in the system and obtain their L2 cache size.
2. CUDA provides cudaGetDeviceCount to find how many devices are available.
3. For each device index, we can use cudaGetDeviceProperties to fill a cudaDeviceProp struct.
4. The struct contains the field l2CacheSize (in bytes). We need to convert this value to megabytes (divide by 1024^2).
5. Output the device index and its L2 cache size in a human‑readable format.
6. Handle error cases: if there are no devices or if any CUDA call fails, print a relevant error message.
7. Keep the program simple, self‑contained, and compileable with nvcc. No external dependencies other than the CUDA runtime.
8. Since the user only wants the program file (.cu), the output will be a single .cu file with a header comment containing the aim and thinking, followed by the complete C/CUDA code.

Potential pitfalls:
- l2CacheSize can be zero on some devices that don't expose an L2 cache size; handle this gracefully.
- Use proper error checking after each CUDA API call.
- Ensure we cast the size division to double or float for fractional MB output, but as the size is typically an integer multiple of KB, integer division suffices for MB, but we can output with decimals for clarity.

Implementation details:
- Use printf for output.
- Use cudaDeviceProp.l2CacheSize.
- Convert bytes to MB: (double)bytes / (1024 * 1024).
- Include <cuda_runtime.h> and <stdio.h>.
- Compile with: nvcc -o device_l2_size device_l2_size.cu
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties for device %d failed: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }

        // Convert bytes to megabytes
        double l2SizeMB = (double)prop.l2CacheSize / (1024.0 * 1024.0);
        printf("Device %d: L2 Cache Size = %.2f MB\n", dev, l2SizeMB);
    }

    return 0;
}
```