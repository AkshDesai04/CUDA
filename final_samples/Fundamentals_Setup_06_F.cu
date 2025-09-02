/*
Print the maximum z-dimension of a grid of blocks.
I want to write a CUDA C program that, when executed, queries the CUDA runtime for the device properties
of the first GPU (device 0). The `cudaDeviceProp` structure contains an array `maxGridSize[3]`
that indicates the maximum allowed dimensions for a grid of thread blocks. I will specifically
print the value of `maxGridSize[2]`, which corresponds to the z-dimension. 
To make the program robust I’ll include a simple error checking macro that reports CUDA API
errors and exits. The program will:
1. Query the number of CUDA-capable devices.
2. Select device 0 (or exit if none exist).
3. Retrieve its properties.
4. Print `prop.maxGridSize[2]`.
The output will be a single line showing the maximum z-dimension value. 
This satisfies the requirement to print the maximum z-dimension of a grid of blocks.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro to check CUDA API call results */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            return 1;                                                       \
        }                                                                   \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }

    int device = 0; /* Use the first device */
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Maximum z-dimension of grid of blocks: %d\n", prop.maxGridSize[2]);

    return 0;
}
