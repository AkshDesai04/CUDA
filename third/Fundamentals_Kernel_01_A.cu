/*
Aim of the program: Modify the program to launch the kernel with two blocks, each with one thread.

Thinking:
1. The requirement is to adjust a CUDA program so that its kernel launch uses 2 blocks and 1 thread per block.
2. We'll define a simple kernel that prints a message indicating the block and thread indices.
3. The kernel will be launched with grid dimension of 2 (blocks) and block dimension of 1 (thread).
4. After the launch, we call cudaDeviceSynchronize() to ensure the kernel completes before the host exits.
5. We include necessary headers, handle basic error checking for kernel launch, and keep the code minimal and self-contained.
6. The program will be compiled with nvcc and produce a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that prints block and thread indices
__global__ void printKernel()
{
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
    // Launch kernel with 2 blocks, 1 thread per block
    printKernel<<<2, 1>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel execution: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
