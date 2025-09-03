/*
Aim: Modify the kernel so that blocks with `blockIdx.x == blockIdx.y` (on the diagonal) print a special message.

Thinking:
The task is to create a CUDA program where a kernel identifies when it is running in a block that lies on the diagonal of a 2D grid (i.e., the block indices in the x and y dimensions are equal). In that case, the kernel should output a special message to the console; otherwise, it can print a normal message indicating its block position.

To satisfy the requirement we:
1. Define a simple kernel that uses `printf` to output information about each block.
2. Inside the kernel we test the condition `blockIdx.x == blockIdx.y`. If true, we print a special message (“Diagonal block!”), otherwise we print the block coordinates.
3. In the host code we launch a grid of multiple blocks arranged in a 2D configuration (e.g., 4×4). Each block will run the same kernel; only the diagonal ones will print the special message.
4. After the kernel launch we synchronize the device and check for errors.

The entire program is written in a single .cu file, uses only standard CUDA APIs, and can be compiled with `nvcc`. It contains only the code – no additional output or explanatory text.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printBlockInfo()
{
    // Each block prints its own index.
    if (blockIdx.x == blockIdx.y)
    {
        printf("Diagonal block! (x=%d, y=%d)\n", blockIdx.x, blockIdx.y);
    }
    else
    {
        printf("Block (x=%d, y=%d)\n", blockIdx.x, blockIdx.y);
    }
}

int main()
{
    // Define a 2D grid of blocks (e.g., 4x4) and 1 thread per block.
    dim3 gridDim(4, 4, 1);
    dim3 blockDim(1, 1, 1);

    // Launch the kernel.
    printBlockInfo<<<gridDim, blockDim>>>();

    // Wait for the GPU to finish.
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
