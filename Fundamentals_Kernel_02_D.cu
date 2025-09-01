/*
Modify the `printf` to include the block's ID: `printf("Hello from block %d!\n", blockIdx.x);`. Launch with 8 blocks.

The goal of this program is to demonstrate a simple CUDA kernel that prints a message from each block. 
We want to modify the existing printf to display the block index (`blockIdx.x`). 
The kernel will be launched with 8 blocks, each containing a single thread. 
Each thread will execute the printf statement and output its block ID. 
After kernel launch we will synchronize and perform error checking to ensure that the kernel executed correctly.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                          \
    } while (0)

__global__ void helloFromBlock()
{
    // Print a greeting that includes the block's ID
    printf("Hello from block %d!\n", blockIdx.x);
}

int main()
{
    // Define grid and block dimensions
    dim3 blockDim(1);  // one thread per block
    dim3 gridDim(8);   // launch 8 blocks

    // Launch the kernel
    helloFromBlock<<<gridDim, blockDim>>>();

    // Wait for the kernel to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset device (optional but good practice)
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
