/*
Based on the `maxThreadsPerBlock` value and a warp size of 32, calculate and print the maximum number of warps per block.

My thinking:
1. We need to query the GPU for its maximum threads per block (`maxThreadsPerBlock`) using the CUDA runtime API.
2. The warp size on most NVIDIA GPUs is 32, but we can also retrieve it from the device properties (`prop.warpSize`). For clarity and to match the requirement, we'll use the constant 32.
3. The number of warps per block is simply `maxThreadsPerBlock / warpSize`. Since `maxThreadsPerBlock` is typically a multiple of 32, integer division will yield the exact number of warps.
4. We'll write a small host program that obtains the device properties, performs the calculation, and prints the results.
5. The program will handle potential errors from the CUDA API calls.
6. No kernels are launched; we only need to perform a device query and print the computed value.

The code below implements this logic in a single .cu file. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0; // Use the first GPU
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    const int warpSize = 32; // specified warp size
    int maxWarpsPerBlock = maxThreadsPerBlock / warpSize;

    printf("Device: %s\n", prop.name);
    printf("Maximum threads per block: %d\n", maxThreadsPerBlock);
    printf("Warp size: %d\n", warpSize);
    printf("Maximum number of warps per block: %d\n", maxWarpsPerBlock);

    return 0;
}
