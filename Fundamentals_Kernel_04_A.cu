/*
Also print the grid's dimension, `gridDim.x`, alongside the block index.

The task is to write a CUDA program that prints each block's index along with the dimension of the grid in the x-direction (`gridDim.x`). The program will:
1. Define a simple kernel that uses `printf` to output the block index and `gridDim.x`.
2. Launch the kernel with a specified number of blocks and threads per block.
3. Use host code to set up the launch parameters, launch the kernel, and synchronize.
4. Include basic error checking for CUDA API calls.

The kernel will run on each block; we only need to print once per block to avoid excessive output. We'll use `blockIdx.x` as the block index. The host code will set `gridDim.x` to 4 and `blockDim.x` to 8 for demonstration purposes. The output will look something like:
```
Block index: 0, GridDim.x: 4
Block index: 1, GridDim.x: 4
Block index: 2, GridDim.x: 4
Block index: 3, GridDim.x: 4
```
This demonstrates the requested information. No external libraries are required beyond the CUDA runtime. The code is fully self‑contained and can be compiled with `nvcc` to produce an executable.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the block index and the grid dimension in x
__global__ void printBlockInfo()
{
    // Only one thread per block does the printing to keep output readable
    if (threadIdx.x == 0)
    {
        printf("Block index: %d, GridDim.x: %d\n", blockIdx.x, gridDim.x);
    }
}

int main(void)
{
    // Define grid and block dimensions
    const int threadsPerBlock = 8;
    const int blocksPerGrid   = 4;

    // Launch kernel
    printBlockInfo<<<blocksPerGrid, threadsPerBlock>>>();

    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Optional: reset device (clean up)
    cudaDeviceReset();

    return 0;
}
