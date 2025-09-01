/*
Aim of the program:
Launch with a 1D grid of 8 blocks, but give each block 4 threads. Have only thread 0 of each block print the `blockIdx.x`.

My thinking:
To satisfy the requirement, I will write a simple CUDA program in C that launches a kernel with 8 blocks in a 1D grid, each block containing 4 threads. Inside the kernel, I will check if the thread index `threadIdx.x` is 0; if so, that thread will use `printf` to output the block index `blockIdx.x`. After launching the kernel from the host, I will call `cudaDeviceSynchronize()` to ensure that all prints are flushed before the program exits. The code will include necessary headers and compile as a `.cu` file. The kernel launch configuration will be `<<<8, 4>>>`. No other output will be produced.*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printBlockIdx()
{
    if (threadIdx.x == 0)
    {
        printf("BlockIdx.x = %d\n", blockIdx.x);
    }
}

int main(void)
{
    // Launch kernel with 8 blocks (grid) and 4 threads per block
    printBlockIdx<<<8, 4>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
