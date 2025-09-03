```cuda
/*
Aim of the program:
Iterate through all devices on the system and print the `asyncEngineCount` for each.

Thought process:
1. We need to query the number of CUDA-capable devices on the system. This is done via `cudaGetDeviceCount`.
2. For each device index from 0 to count-1, we need to retrieve its properties. Use `cudaGetDeviceProperties` which fills a `cudaDeviceProp` struct.
3. The property we are interested in is `asyncEngineCount`. This indicates the number of asynchronous engine resources available on the device.
4. Print the device index, name, and asyncEngineCount to the console. We can use `printf` for simplicity.
5. Handle possible errors: if `cudaGetDeviceCount` or `cudaGetDeviceProperties` returns an error, print a message and exit.
6. The program should compile as a .cu file, so we include `<cuda_runtime.h>` and write a `main` function. No device kernels are needed.
7. Since the requirement is to write the entire code from start to end, we will include all necessary includes and the main function, and ensure proper return values.

Implementation notes:
- Use a simple error-checking helper macro to wrap CUDA calls.
- Output format: "Device X: Name = ..., Async Engine Count = Y".
- The program is self-contained and does not require any external files.

This satisfies the requirement of iterating over all devices and printing the asyncEngineCount for each.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Helper macro for error checking
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA-capable device(s):\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: Name = %s, Async Engine Count = %d\n",
               dev, prop.name, prop.asyncEngineCount);
    }

    return EXIT_SUCCESS;
}
```