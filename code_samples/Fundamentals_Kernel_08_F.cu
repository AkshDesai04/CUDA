/*
Launch a 4x4 grid. Have only the block at (0, 0) print a message.

Thinking process:
- The requirement is to launch a 4x4 grid of blocks. This means we need to set grid dimensions to dim3(4,4).
- Only the block with indices (0,0) should print. All other blocks should do nothing.
- We will use a simple kernel that checks blockIdx.x and blockIdx.y. If both are zero, it prints a message using printf.
- Since we want a single thread per block, block dimensions can be (1,1,1). The number of threads per block doesn't matter for this example, but using one keeps the code minimal.
- After launching the kernel, we need to synchronize the device to ensure the output is flushed before the program exits. 
- Basic error checking after kernel launch is not strictly required by the prompt, but we could include cudaDeviceSynchronize() and check its return value. For brevity, we simply call cudaDeviceSynchronize().
- The program is written in CUDA C and will be compiled with nvcc. The file extension is .cu.
- The output will be printed to the console by the block (0,0).
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printKernel()
{
    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        printf("Hello from block (0,0)!\n");
    }
}

int main()
{
    dim3 gridDim(4, 4);   // Launch a 4x4 grid
    dim3 blockDim(1, 1);  // One thread per block

    printKernel<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();

    return 0;
}
