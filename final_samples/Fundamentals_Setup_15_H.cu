/*
Write a loop that iterates through all available devices and calls `cudaSetDevice` followed by `cudaDeviceReset` for each one.

Thought Process:
- First, we need to query the number of CUDA-capable devices using cudaGetDeviceCount().
- For each device index from 0 to count-1, we set the device with cudaSetDevice().
- After setting the device, we reset it with cudaDeviceReset() to ensure it returns to its initial state.
- It's prudent to check the return status of each CUDA API call and report any errors.
- We'll output simple console messages indicating the progress and any errors encountered.
- The program will be a self-contained .cu file that can be compiled with nvcc.
- Since the logic is straightforward, no GPU kernels are launched; we only interact with the driver API.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        printf("\nSetting device %d...\n", dev);
        err = cudaSetDevice(dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Resetting device %d...\n", dev);
        err = cudaDeviceReset();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset() failed on device %d: %s\n", dev, cudaGetErrorString(err));
        } else {
            printf("Device %d reset successfully.\n", dev);
        }
    }

    return EXIT_SUCCESS;
}
