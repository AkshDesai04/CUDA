```c
/*
Aim of the program: Query and print the maximum number of registers available per block.

Thought process:
- We want to use CUDA Runtime API to retrieve device properties.
- The attribute for maximum registers per block is cudaDevAttrMaxRegistersPerBlock.
- We can get the current device using cudaGetDevice, then use cudaDeviceGetAttribute to fetch the attribute.
- For robustness, we will also handle errors and print a friendly message if the query fails.
- The program will be a minimal console application that prints the result and exits.
- The code will include necessary headers, perform the query, print the result, and return an appropriate status.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int maxRegsPerBlock = 0;
    err = cudaDeviceGetAttribute(&maxRegsPerBlock,
                                 cudaDevAttrMaxRegistersPerBlock,
                                 device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to query MaxRegistersPerBlock: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum registers available per block on device %d: %d\n",
           device, maxRegsPerBlock);

    return 0;
}
```