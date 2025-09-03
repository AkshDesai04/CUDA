/*
Launch a 4x4 grid. Have only the block at (0, 0) print a message.

Thinking process:
- The problem requires a CUDA program that launches a 4x4 grid of blocks.
- We want only the block with indices (0,0) to print a message. All other blocks should do nothing.
- In CUDA, the block indices are accessible via blockIdx.x and blockIdx.y inside a kernel.
- We can use the device printf function to print from the kernel. To ensure the output appears, we must synchronize the device after launching the kernel.
- For simplicity, each block can have a single thread (blockDim = 1,1,1). 
- The kernel will check if blockIdx.x == 0 && blockIdx.y == 0 and call printf only in that case.
- After the kernel launch, we call cudaDeviceSynchronize() in the host code to flush the output before the program exits.
- The code will include necessary headers (<stdio.h> for printf and CUDA runtime APIs).
- The compile command would be something like: nvcc -o program program.cu
- The code below follows these decisions.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void launchGridKernel()
{
    // Only the block at (0,0) prints the message
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Hello from block (0,0)!\n");
    }
}

int main()
{
    // Define a 4x4 grid of blocks, each with a single thread
    dim3 gridDim(4, 4);
    dim3 blockDim(1, 1);

    // Launch the kernel
    launchGridKernel<<<gridDim, blockDim>>>();

    // Ensure the device has finished executing and flush printf output
    cudaDeviceSynchronize();

    return 0;
}
