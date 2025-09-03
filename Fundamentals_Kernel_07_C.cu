/*
Also print the block's 2D dimensions (`blockDim.x`, `blockDim.y`).

My thinking:

- The goal is to write a simple CUDA program that launches a kernel and prints out the 2D dimensions of the block (`blockDim.x` and `blockDim.y`). 
- To see the output from the device, we use `printf` inside the kernel. This requires the device to support printf; modern CUDA devices do, but the compilation flag `-arch=sm_20` or higher is typically enough. 
- We’ll launch a single block with a 2D block size (e.g., 8x4). Inside the kernel we’ll print the dimensions only from a single thread (e.g., threadIdx.x == 0 && threadIdx.y == 0) to avoid duplicate prints. 
- The host code will set up the launch configuration, launch the kernel, and synchronize. 
- Basic error checking is added for the kernel launch and for `cudaDeviceSynchronize`. 
- The code is self‑contained and ready to compile with `nvcc -arch=sm_20 -o print_block_dims print_block_dims.cu`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints blockDim.x and blockDim.y from a single thread
__global__ void printBlockDims()
{
    // Only the first thread in each block prints to avoid flooding the output
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        printf("Block dimensions: x = %d, y = %d\n", blockDim.x, blockDim.y);
    }
}

int main()
{
    // Define block and grid dimensions
    dim3 blockDim(8, 4);   // 8 threads in x, 4 threads in y
    dim3 gridDim(1, 1);    // single block

    // Launch the kernel
    printBlockDims<<<gridDim, blockDim>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
