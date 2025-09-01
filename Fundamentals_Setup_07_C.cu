```c
/*
Write a program that prints a summary of the execution hierarchy limits: Max Grid [X, Y, Z] and Max Block [X, Y, Z].

Thinking:
- The goal is to query CUDA device properties using the runtime API.
- Use cudaGetDevice() to get the current device.
- Use cudaGetDeviceProperties() to retrieve the struct cudaDeviceProp for that device.
- The struct contains arrays maxGridSize[3] for grid dimensions and maxThreadsDim[3] for block dimensions.
- Print these values in a readable format.
- Include basic error checking for CUDA calls.
- Compile with nvcc, which expects a .cu file.
- The program is simple, single-threaded, and self-contained.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int device = 0;
    cudaError_t err;

    // Get the current device
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print device name
    printf("Device %d: %s\n", device, prop.name);

    // Print maximum grid dimensions
    printf("Max Grid  [X, Y, Z]: [%d, %d, %d]\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    // Print maximum block dimensions
    printf("Max Block [X, Y, Z]: [%d, %d, %d]\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

    // Print maximum threads per block (optional)
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);

    return 0;
}
```