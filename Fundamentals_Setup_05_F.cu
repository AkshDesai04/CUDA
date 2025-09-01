/*
Write a program that prints all key resource limits for a block: max threads, max shared memory, and max registers.
[Thinking:
- We need to query the device properties from CUDA runtime API.
- The relevant fields are:
    - prop.maxThreadsPerBlock for maximum threads per block.
    - prop.sharedMemPerBlock for maximum shared memory per block (in bytes).
    - prop.regsPerBlock for maximum registers per block.
- We will get the current device, query its properties, and print these values.
- Basic error checking will be added for CUDA API calls.
- The program will compile with nvcc and produce a simple console output.
]
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("CUDA Device %d: %s\n", device, prop.name);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max shared memory per block: %d bytes\n", prop.sharedMemPerBlock);
    printf("  Max registers per block: %d\n", prop.regsPerBlock);

    return 0;
}
