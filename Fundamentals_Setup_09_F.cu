/*
Aim: Check if the device supports stream priorities (streamPrioritiesSupported).

Thinking:
1. The goal is to inspect each CUDA-capable device on the system and determine
   whether it supports stream priorities. CUDA exposes this information via the
   cudaDeviceProp struct, specifically the `streamPrioritiesSupported` member.
2. The program will:
   - Query the number of CUDA devices with cudaGetDeviceCount().
   - Iterate over each device index, retrieve its properties using
     cudaGetDeviceProperties().
   - Inspect the `streamPrioritiesSupported` field and print a user-friendly
     message indicating whether stream priorities are supported on that device.
3. Error handling:
   - All CUDA runtime calls will be checked for errors. If an error occurs,
     the program will print an error message and exit.
   - If no devices are found, the program will inform the user.
4. The code is written in standard C (C99) style with minimal dependencies,
   making it straightforward to compile with nvcc.
5. The output is simple console text, e.g.:
      Device 0: GeForce GTX 1080
        Stream priorities supported: Yes
   This satisfies the requirement of checking support for stream priorities.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
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
            fprintf(stderr, "Error: cudaGetDeviceProperties for device %d failed: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }

        printf("\nDevice %d: %s\n", dev, prop.name);
        printf("  Stream priorities supported: %s\n",
               prop.streamPrioritiesSupported ? "Yes" : "No");
    }

    return EXIT_SUCCESS;
}
