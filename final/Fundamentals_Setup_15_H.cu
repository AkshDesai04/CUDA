```cuda
/*
Write a loop that iterates through all available devices and calls `cudaSetDevice` followed by `cudaDeviceReset` for each one.

Thinking:
1. The goal is to iterate over every CUDA-capable device present in the system.
2. First, obtain the number of devices with cudaGetDeviceCount().
3. Loop from 0 to count-1:
   - Call cudaSetDevice(i) to select the device.
   - After setting the device, immediately call cudaDeviceReset() to reset it.
4. Add basic error checking after each CUDA API call to handle failures gracefully.
5. Print messages to indicate progress and any errors.
6. The program uses the CUDA Runtime API, includes <cuda_runtime.h>, and compiles as a .cu file.
7. No external libraries or extra output are required. The program simply performs the loop and exits.
*/
#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        printf("\n--- Device %d ---\n", dev);

        err = cudaSetDevice(dev);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev, cudaGetErrorString(err));
            continue;  // Skip to next device
        }
        printf("Set device to %d successfully.\n", dev);

        err = cudaDeviceReset();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaDeviceReset() failed on device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }
        printf("Device %d reset successfully.\n", dev);
    }

    printf("\nAll devices processed.\n");
    return EXIT_SUCCESS;
}
```