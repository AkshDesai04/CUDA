/*
Aim: In the deadlock example, threads < 16 reach the barrier, but threads >= 16 do not. The first group waits forever for the second group, which will never arrive.

Thinking:
The goal is to create a simple CUDA kernel that demonstrates a classic deadlock scenario caused by an improper use of __syncthreads(). In CUDA, __syncthreads() is a barrier that synchronizes all threads within a block. If only a subset of the threads execute this barrier, the kernel will hang because the missing threads never reach the barrier and thus the thread that did call __syncthreads() waits indefinitely. This is a common mistake when writing conditional code that inadvertently skips the barrier for some threads.

To illustrate this, the kernel will:
1. Compute a unique thread id.
2. Print a message before the barrier.
3. Only threads with id < 16 will call __syncthreads(); threads with id >= 16 will skip it.
4. After the barrier, all threads will attempt to print a message.

Because the first 16 threads will wait on __syncthreads() and the remaining threads will never reach it, the kernel will never terminate, causing the host program to block on cudaDeviceSynchronize().

This example uses device printf (available on compute capability >= 2.0). The host code launches a single block of 32 threads, starts the kernel, and then synchronizes. The synchronization will block indefinitely, demonstrating the deadlock.

Note: Running this program will cause the host process to hang. Terminate it manually if needed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void deadlockKernel()
{
    // Compute global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Print before barrier
    printf("Thread %d: Before barrier\n", tid);

    // Only a subset of threads calls __syncthreads()
    if (tid < 16)
    {
        __syncthreads(); // Barrier that will never be satisfied
    }

    // This line will never be reached for threads that hit the barrier
    printf("Thread %d: After barrier\n", tid);
}

int main()
{
    // Define launch configuration
    const int threadsPerBlock = 32;
    const int numberOfBlocks  = 1;

    // Launch the kernel
    deadlockKernel<<<numberOfBlocks, threadsPerBlock>>>();

    // Synchronize and check for errors
    cudaError_t err = cudaDeviceSynchronize();

    // This point will never be reached due to the deadlock
    if (err != cudaSuccess)
    {
        printf("Kernel finished with error: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("Kernel finished successfully.\n");
    }

    return 0;
}
