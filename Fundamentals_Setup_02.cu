```cpp
/*
 * Aim of the program
 * Extend the previous program to iterate through all available CUDA devices.
 * For each device, use `cudaGetDeviceProperties` to retrieve its properties and print its name
 * (e.g., 'NVIDIA GeForce RTX 3080') and its compute capability
 * (e.g., 'Compute Capability: 8.6').
 *
 * Thinking process:
 * 1. Query the number of CUDA-capable devices with cudaGetDeviceCount().
 * 2. Iterate over each device index from 0 to deviceCount-1.
 * 3. For each device, call cudaGetDeviceProperties() to obtain a cudaDeviceProp structure.
 * 4. From the structure, extract:
 *    - prop.name for the device name.
 *    - prop.major and prop.minor for the compute capability.
 * 5. Print the information in a readable format.
 * 6. Include basic error checking after each CUDA API call, printing error messages
 *    and exiting if an error occurs. This ensures robustness when run on systems
 *    with zero or multiple GPUs.
 * 7. The program is simple, uses only the CUDA Runtime API and standard C++ I/O
 *    functions (printf), making it portable and compilable with nvcc.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s (%s)\n",                \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err));     \
            return EXIT_FAILURE;                                             \
        }                                                                     \
    } while (0)

int main(void) {
    int deviceCount = 0;
    /* Get the number of CUDA devices */
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Number of CUDA-capable devices: %d\n", deviceCount);

    /* Iterate over all devices */
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        /* Retrieve device properties */
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        /* Print device name and compute capability */
        printf("\nDevice %d:\n", dev);
        printf("  Name: %s\n", prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    }

    return EXIT_SUCCESS;
}
```