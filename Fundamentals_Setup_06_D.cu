/*
Print the maximum x-dimension of a grid of blocks.

Thinking: To find the maximum x‑dimension of a grid of blocks in CUDA, we query the device properties using cudaGetDeviceProperties. The device property struct contains a member maxGridSize[0] which holds the maximum grid dimension in the x direction for the current device. The program will:
1. Retrieve the current device number.
2. Obtain the cudaDeviceProp structure for that device.
3. Print the value of maxGridSize[0].
Error handling is added to catch any failures from the CUDA runtime API calls.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current device (%d): %s\n", err, cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties (%d): %s\n", err, cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum x-dimension of a grid of blocks: %d\n", prop.maxGridSize[0]);

    return 0;
}
