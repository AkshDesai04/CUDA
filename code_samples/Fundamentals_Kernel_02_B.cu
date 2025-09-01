/*
Aim: Launch the printing kernel with 10 blocks of one thread. Observe the output.

Thinking:
The goal is to demonstrate how to launch a CUDA kernel that uses printf to output information from device code. 
We want 10 blocks, each with a single thread, so that the output will show 10 lines, one per block. 
The kernel will print its block index and thread index. After launching, we must synchronize the device 
to ensure all printf calls are completed before the program exits. Error checking after the launch and 
synchronization is included for robustness. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that prints block and thread indices
__global__ void printKernel()
{
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

// Utility macro for error checking
#define CHECK_CUDA(call)                                   \
    do {                                                   \
        cudaError_t err = call;                            \
        if (err != cudaSuccess) {                          \
            fprintf(stderr, "CUDA error in %s:%d: %s\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                            \
        }                                                  \
    } while (0)

int main(void)
{
    printf("Launching kernel with 10 blocks, 1 thread each...\n");

    // Launch kernel
    printKernel<<<10, 1>>>();
    CHECK_CUDA(cudaGetLastError());    // Check for launch errors

    // Synchronize to ensure all printf output is flushed
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Kernel execution completed.\n");

    // Clean up device (optional but good practice)
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
