/*
Aim: Launch with a 3D grid of (2, 2, 2) blocks, each with one thread. Have each thread print its `blockIdx.x`, `blockIdx.y`, and `blockIdx.z`.

Thinking: 
To satisfy the requirement, we need a CUDA kernel that prints the block indices of each thread.  
Since each block contains only one thread, the kernel will be launched with a block dimension of (1,1,1).  
The grid dimension should be (2,2,2) as specified.  
We use `printf` inside the kernel, which is supported on devices with compute capability 2.0 and higher.  
The host code will set up the grid and block dimensions, launch the kernel, and synchronize to ensure all output is flushed before the program exits.  
No additional device memory or synchronization primitives are needed because the operation is embarrassingly simple.  
The output will show eight lines, one for each block, displaying the block indices.  
*/

#include <stdio.h>

__global__ void printBlockIdx(void)
{
    // Each thread prints its block index
    printf("BlockIdx: (%d, %d, %d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z);
}

int main(void)
{
    // Define a 3D grid of (2, 2, 2) blocks
    dim3 gridDim(2, 2, 2);
    // Each block has a single thread
    dim3 blockDim(1, 1, 1);

    // Launch the kernel
    printBlockIdx<<<gridDim, blockDim>>>();

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    return 0;
}
