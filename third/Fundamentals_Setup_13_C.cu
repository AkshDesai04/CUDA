```c
/*
Aim: Iterate through all devices and print the clock rate for each one.

Thinking:
1. Use the CUDA Runtime API to discover how many CUDA-capable devices are present.
2. For each device index, obtain its properties via cudaGetDeviceProperties().
3. The clock rate is stored in the deviceProperties.clockRate field, which is in kHz.
4. Convert to MHz for a more readable output (divide by 1000).
5. Handle potential errors from the CUDA API calls by printing a message and exiting.
6. Include necessary headers: <cuda_runtime.h> and <stdio.h>.
7. The program will be compiled with nvcc, producing a .cu file that can be run to display the
   clock rates of all detected GPUs.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA-capable device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error getting properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue; // skip to next device
        }

        // clockRate is in kHz; convert to MHz for readability
        double clockMHz = prop.clockRate / 1000.0;

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Clock Rate: %.2f MHz\n", clockMHz);
    }

    return EXIT_SUCCESS;
}
```