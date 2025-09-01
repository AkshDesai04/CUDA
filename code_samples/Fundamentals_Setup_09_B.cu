```cuda
/*
Aim of the program: Query and print the `asyncEngineCount` property, which indicates the number of engines available for overlapping data copies and kernel execution.

Thinking:
1. Use the CUDA runtime API to obtain the number of CUDA-capable devices via cudaGetDeviceCount.
2. For each device, retrieve the device properties using cudaGetDeviceProperties.
3. Extract the asyncEngineCount field from the cudaDeviceProp structure.
4. Print the device index, name, and asyncEngineCount to standard output.
5. Include error checking after each CUDA API call to handle potential failures gracefully.
6. The program is written in C with CUDA extensions (.cu) and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error retrieving properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("\nDevice %d: \"%s\"\n", dev, prop.name);
        printf("  asyncEngineCount: %d\n", prop.asyncEngineCount);
    }

    return 0;
}
```