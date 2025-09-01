/*
Launch with a 1D grid of 8 blocks, but give each block 4 threads. Have only thread 0 of each block print the `blockIdx.x`.

The goal is to write a minimal CUDA program that launches a kernel with a 1D grid consisting of 8 blocks, each containing 4 threads. Inside the kernel, we only want thread 0 of each block to perform a `printf` of the block index (`blockIdx.x`). This demonstrates basic kernel launch parameters, intra-block thread identification, and device-side printing. The program includes standard error checking after the kernel launch and a device synchronization to ensure all output is flushed before the program exits. The device `printf` requires compute capability ≥ 2.0, which is typically satisfied on most CUDA-capable GPUs.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printBlockIdx()
{
    if (threadIdx.x == 0)
    {
        printf("blockIdx.x = %d\n", blockIdx.x);
    }
}

int main()
{
    dim3 grid(8);
    dim3 block(4);

    printBlockIdx<<<grid, block>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after device synchronize: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
