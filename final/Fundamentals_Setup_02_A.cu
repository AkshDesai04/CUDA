```c
/*
 * Aim of the program (word for word):
 * "Iterate through all available CUDA devices and print their total global memory in gigabytes (GiB) for each."
 *
 * Thought process:
 * 1. We need to use the CUDA Runtime API to query device information. The primary functions are:
 *    - cudaGetDeviceCount(&count): obtains the number of CUDA-capable devices.
 *    - cudaGetDeviceProperties(&props, device): returns a cudaDeviceProp struct containing device characteristics.
 * 2. The total global memory is provided in the `totalGlobalMem` field of `cudaDeviceProp`. This is expressed in bytes.
 * 3. We want to print this memory in GiB (gibibytes), which is 1024^3 bytes. So conversion factor = 1024 * 1024 * 1024.
 * 4. We need to handle potential errors: if `cudaGetDeviceCount` or `cudaGetDeviceProperties` return non-`cudaSuccess`, we should print an error and exit.
 * 5. The program will be a simple `main` function that:
 *    - Calls `cudaGetDeviceCount`.
 *    - Loops over device indices.
 *    - For each device, retrieves properties and prints: "Device <i>: <name>, Total Global Memory: <GiB> GiB".
 * 6. To format floating point nicely, we can use `printf` with `%.2f` for two decimal places.
 * 7. We will include `<cuda_runtime.h>` for CUDA types and functions, `<stdio.h>` for I/O.
 * 8. Since this is a CUDA host program, we compile it with `nvcc` and it will be a normal executable.
 * 9. No GPU kernel is required; everything runs on the host.
 *
 * Edge cases considered:
 * - No CUDA devices present: print a message and exit gracefully.
 * - Very large memory values: double should be sufficient for GiB conversion.
 * - Error checking ensures that any CUDA runtime errors are reported.
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

    printf("Detected %d CUDA device(s):\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error getting properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        // Convert totalGlobalMem from bytes to GiB
        double memGiB = (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Total Global Memory: %.2f GiB\n", memGiB);
    }

    return EXIT_SUCCESS;
}
```