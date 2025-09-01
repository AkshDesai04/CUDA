```c
/*
Aim of the program: Print the maximum number of resident blocks per multiprocessor.

Thinking:
1. The goal is to query CUDA device properties to obtain the value for the maximum resident blocks that can be scheduled on a single multiprocessor (SM) at a time. This value is stored in the cudaDeviceProp structure field `maxBlocksPerMultiprocessor`.
2. To do this, we use the CUDA Runtime API. Specifically, we call `cudaGetDeviceProperties(&prop, dev)` to fill a `cudaDeviceProp` struct for a given device index (we use 0 for the first device).
3. Once the properties are retrieved, we simply print the field `prop.maxBlocksPerMultiprocessor` using standard C I/O (`printf`).
4. We also perform basic error checking: if `cudaGetDeviceProperties` returns a non-zero error code, we print an error message and exit.
5. The program is intended to be compiled with `nvcc`. Therefore we include `<cuda_runtime.h>` for CUDA runtime functions and `<stdio.h>` for printing.
6. No other supporting code or output is required; the program prints only the requested value.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    cudaDeviceProp prop;
    int device = 0; // Use the first CUDA device

    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return -1;
    }

    printf("Maximum resident blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiprocessor);

    return 0;
}
```