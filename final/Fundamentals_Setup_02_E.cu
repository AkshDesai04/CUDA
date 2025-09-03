/*
Write a program that finds the device with the most global memory and prints its name and index.

Thinking:
1. Include necessary CUDA headers and standard I/O libraries.
2. Use cudaGetDeviceCount() to retrieve the number of CUDA-capable devices.
3. Iterate over each device index:
   a. Retrieve device properties with cudaGetDeviceProperties().
   b. Access the globalMemory field to determine the amount of global memory.
   c. Keep track of the maximum memory seen so far and store the corresponding device index and name.
4. After the loop, print the device name and index that has the most global memory.
5. Handle error cases: if no devices are found, print an appropriate message.
6. Use standard output (printf) for simplicity and portability.
7. Compile with nvcc, e.g., `nvcc -o max_mem_device max_mem_device.cu`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    size_t maxGlobalMem = 0;
    int maxIndex = -1;
    char maxName[256] = {0};

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        // Compare global memory sizes
        if (prop.totalGlobalMem > maxGlobalMem)
        {
            maxGlobalMem = prop.totalGlobalMem;
            maxIndex = dev;
            strncpy(maxName, prop.name, sizeof(maxName) - 1);
            maxName[sizeof(maxName) - 1] = '\0'; // Ensure null-terminated
        }
    }

    if (maxIndex >= 0)
    {
        printf("Device with the most global memory:\n");
        printf("Index: %d\n", maxIndex);
        printf("Name : %s\n", maxName);
        printf("Global Memory: %zu bytes (%.2f GB)\n",
               maxGlobalMem,
               (double)maxGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    else
    {
        printf("Could not determine the device with the most global memory.\n");
    }

    return EXIT_SUCCESS;
}
