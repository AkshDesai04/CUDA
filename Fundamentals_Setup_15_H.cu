/*
Write a loop that iterates through all available devices and calls `cudaSetDevice` followed by `cudaDeviceReset` for each one.

Thought process:
- The program should query how many CUDA devices are present with `cudaGetDeviceCount`.
- It should iterate from 0 to deviceCount-1.
- For each device index, first call `cudaSetDevice(i)` to make that device the current one.
- Then call `cudaDeviceReset()` which resets the device and cleans up resources.
- Error checking is important: check return status of both CUDA API calls and report failures.
- Use standard C/C++ output to indicate progress.
- Wrap the logic in `int main()` returning 0 on success.
- Include necessary headers: <cuda_runtime.h> and <cstdio> or <iostream>.
- Keep the code minimal but robust, and compileable with `nvcc`.
- No additional text or explanation outside the code is required.
*/

#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        printf("Setting device %d\n", dev);
        err = cudaSetDevice(dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev, cudaGetErrorString(err));
            // Continue to next device to attempt resetting others
            continue;
        }

        printf("Resetting device %d\n", dev);
        err = cudaDeviceReset();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset() on device %d failed: %s\n", dev, cudaGetErrorString(err));
            // Continue to next device
            continue;
        }

        printf("Device %d reset successfully.\n", dev);
    }

    printf("All devices processed.\n");
    return EXIT_SUCCESS;
}
