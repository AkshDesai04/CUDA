```cpp
/*
 * Aim: Write a program that finds the device with the highest compute capability and prints its name and index.
 *
 * Thought Process:
 * 1. Use CUDA runtime API to query the number of devices via cudaGetDeviceCount.
 * 2. Iterate over all devices, retrieving their properties with cudaGetDeviceProperties.
 * 3. Extract the compute capability major and minor numbers (prop.major, prop.minor).
 * 4. Keep track of the device with the highest compute capability by comparing first the major number,
 *    and if equal, comparing the minor number.
 * 5. Store the name and index of the best device found.
 * 6. After the loop, print the best device's name, index, and compute capability.
 * 7. Handle errors gracefully by checking return codes from CUDA API calls and printing readable
 *    messages using cudaGetErrorString.
 * 8. Include necessary headers: <cuda_runtime.h> for CUDA APIs, <stdio.h> for I/O, <stdlib.h> for
 *    EXIT_SUCCESS/FAILURE, and <string.h> for strncpy.
 * 9. The program compiles with nvcc as a standalone .cu file.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0)
    {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    int bestIndex   = -1;
    int bestMajor   = -1;
    int bestMinor   = -1;
    char bestName[256] = {0};

    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        // Determine if this device has a higher compute capability
        if (prop.major > bestMajor ||
            (prop.major == bestMajor && prop.minor > bestMinor))
        {
            bestMajor = prop.major;
            bestMinor = prop.minor;
            bestIndex = i;
            strncpy(bestName, prop.name, sizeof(bestName) - 1);
            bestName[sizeof(bestName) - 1] = '\0';  // Ensure null-termination
        }
    }

    if (bestIndex >= 0)
    {
        printf("Device with highest compute capability:\n");
        printf("  Name: %s\n", bestName);
        printf("  Index: %d\n", bestIndex);
        printf("  Compute Capability: %d.%d\n", bestMajor, bestMinor);
    }
    else
    {
        fprintf(stderr, "Could not determine device with highest compute capability.\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```