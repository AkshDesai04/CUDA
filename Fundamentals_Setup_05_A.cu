/*
Based on the `maxThreadsPerBlock` value and a warp size of 32, calculate and print the maximum number of warps per block.

Thinking:
1. The goal is to create a CUDA C program that queries the current GPU device for its maximum threads per block (maxThreadsPerBlock).
2. With the known warp size of 32 threads, we can compute the maximum number of warps that fit in a single block by integer division: maxWarps = maxThreadsPerBlock / 32.
3. The program should:
   - Initialize CUDA, get the device count, and pick the first device (device 0).
   - Retrieve the device properties using cudaGetDeviceProperties.
   - Extract the maxThreadsPerBlock value.
   - Compute the number of warps per block.
   - Print both the maxThreadsPerBlock and the computed number of warps.
4. Error handling: check the CUDA API return status and exit with a non-zero code if any error occurs.
5. The code will be compiled with nvcc, so it should include <cuda_runtime.h> and use standard C I/O functions for output.
6. Ensure the code is self‑contained, with a single main function and no external dependencies beyond the CUDA runtime.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device detected.\n");
        return 1;
    }

    // Use the first device (device 0)
    int device = 0;
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    const int warpSize = 32;
    int maxWarpsPerBlock = maxThreadsPerBlock / warpSize;

    printf("Device %d (%s)\n", device, prop.name);
    printf("Maximum threads per block: %d\n", maxThreadsPerBlock);
    printf("Warp size (threads per warp): %d\n", warpSize);
    printf("Maximum number of warps per block: %d\n", maxWarpsPerBlock);

    return 0;
}
