```cuda
/*
Aim of the program:
Write a kernel that calculates the global ID and then prints whether it is even or odd.

Thinking:
- The goal is to create a simple CUDA program that demonstrates basic kernel launch and thread identification.
- Each thread calculates its global ID using the standard formula:
    global_id = blockIdx.x * blockDim.x + threadIdx.x
- Based on the parity of this global ID, the thread will print either "Thread X is even" or "Thread X is odd".
- Since the kernel uses `printf`, we must call `cudaDeviceSynchronize()` on the host side to ensure all output from device prints is flushed before the program exits.
- Error checking for CUDA API calls will be included to make the program robust.
- The program will launch a single block with a configurable number of threads to illustrate the concept clearly.
- No device memory is allocated; the program purely demonstrates thread indexing and printing.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints whether the global thread ID is even or odd
__global__ void printParityKernel()
{
    // Calculate the global ID of the thread
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine parity and print accordingly
    if (global_id % 2 == 0)
    {
        printf("Thread %u is even\n", global_id);
    }
    else
    {
        printf("Thread %u is odd\n", global_id);
    }
}

int main()
{
    // Define grid and block dimensions
    const int threadsPerBlock = 8;   // Example: 8 threads per block
    const int blocksPerGrid = 2;     // Example: 2 blocks

    // Launch the kernel
    printParityKernel<<<blocksPerGrid, threadsPerBlock>>>();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Synchronize to ensure all printf statements are flushed
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset device (optional but good practice)
    cudaDeviceReset();

    return 0;
}
```