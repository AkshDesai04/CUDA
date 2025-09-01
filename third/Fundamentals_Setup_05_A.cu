```cpp
/*
Based on the maxThreadsPerBlock value and a warp size of 32, calculate and print the maximum number of warps per block.

Thinking process:
1. We need to query the CUDA device for its maximum threads per block capability. This can be obtained via the cudaGetDeviceProperties function, which fills a cudaDeviceProp struct.
2. The warp size on NVIDIA GPUs is 32 threads. Since we already know this, we can simply divide maxThreadsPerBlock by 32 to get the maximum number of warps that can fit in a block.
3. We will write a simple CUDA-capable C program that:
   - Retrieves the properties of the current device.
   - Calculates the number of warps per block.
   - Prints the result to the console.
4. The program is kept minimal and self-contained; no CUDA kernels are launched because the goal is purely to query and compute a device property.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int device = 0;  // Use the first CUDA device
    cudaDeviceProp prop;

    // Get properties of the device
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device properties: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    const int warpSize = 32;
    int maxWarpsPerBlock = prop.maxThreadsPerBlock / warpSize;

    printf("Device %d: Max threads per block = %d\n", device, prop.maxThreadsPerBlock);
    printf("Maximum number of warps per block (warp size = %d) = %d\n", warpSize, maxWarpsPerBlock);

    return EXIT_SUCCESS;
}
```