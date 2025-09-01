/*
Write a kernel that calculates the global ID and then prints whether it is even or odd.
I decided to create a simple CUDA program that defines a kernel which, for each thread,
calculates its global thread ID using the formula:
    global_id = blockIdx.x * blockDim.x + threadIdx.x
It then checks whether this global ID is even or odd by using the modulo operator
(global_id % 2). Depending on the result, the kernel prints a message from the
device using the device-side printf function. In the host code (main), I set up a
grid and block configuration, launch the kernel, and synchronize the device to
ensure all device-side prints are completed before the program exits. The program
includes basic error checking for kernel launch and device synchronization.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that calculates the global thread ID and prints whether it is even or odd
__global__ void printEvenOddKernel()
{
    // Calculate the global thread ID
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine if the global ID is even or odd
    if (global_id % 2 == 0)
    {
        printf("Thread %d: even\n", global_id);
    }
    else
    {
        printf("Thread %d: odd\n", global_id);
    }
}

int main()
{
    // Define the number of threads per block and the number of blocks
    const int threadsPerBlock = 8;
    const int blocksPerGrid = 4;  // Adjust as needed to see more output

    // Launch the kernel
    printEvenOddKernel<<<blocksPerGrid, threadsPerBlock>>>();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Synchronize the device to ensure all prints are finished
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
