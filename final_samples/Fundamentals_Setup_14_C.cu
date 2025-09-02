```cpp
/*
Iterate through all devices on the system and print the `asyncEngineCount` for each.

Thinking process:
- The task is to enumerate all CUDA-capable devices present in the system.
- CUDA provides `cudaGetDeviceCount` to get the number of devices.
- For each device index, `cudaGetDeviceProperties` fills a `cudaDeviceProp` structure which contains the field `asyncEngineCount`.
- We need to print this value for each device.
- The program will be a simple CUDA C++ host program that uses the CUDA Runtime API.
- Error checking is added for robustness: if any CUDA call fails, the program prints an error message and exits.
- The output is printed to the standard output using `printf`.
- Since this is a .cu file, the main function will compile with nvcc and run on the host.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }
        printf("Device %d (%s): asyncEngineCount = %d\n", dev, prop.name, prop.asyncEngineCount);
    }

    return EXIT_SUCCESS;
}
```