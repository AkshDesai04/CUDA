/*
Aim: Also print the thread's `blockIdx.x` and `threadIdx.x` next to its calculated global ID.

Thinking:
The goal of the program is to demonstrate how to retrieve and print CUDA thread identifiers
(blockIdx.x, threadIdx.x) as well as compute the global thread ID.  The global ID is
computed as `blockIdx.x * blockDim.x + threadIdx.x`.  Each thread will print its own
global ID along with the block and thread indices.  The program will launch a simple
kernel with a small number of blocks and threads for clarity, and it will use the
device `printf` facility.  Error checking is added for the kernel launch and
synchronization.  The code is a complete .cu file that can be compiled with nvcc
(e.g., `nvcc -o thread_ids thread_ids.cu` and run on a CUDA-capable device).
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printThreadInfo()
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Global ID: %d, blockIdx.x: %d, threadIdx.x: %d\n",
           global_id, blockIdx.x, threadIdx.x);
}

int main()
{
    const int threads_per_block = 8;
    const int num_blocks = 4; // Total of 32 threads

    // Launch kernel
    printThreadInfo<<<num_blocks, threads_per_block>>>();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel execution: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
