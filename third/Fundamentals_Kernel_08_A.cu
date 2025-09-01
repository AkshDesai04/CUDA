```cuda
/*
Aim of the program: Change the launch configuration to a non-square 2D grid, like 8x2 blocks, and print the 2D indices.

Thinking process:
1. The user wants a non-square 2D grid launch configuration. I interpret this as launching the kernel with a grid of 8 blocks in the x-dimension and 2 blocks in the y-dimension.
2. The kernel should print the 2D indices. I decided to print both the block indices (blockIdx.x, blockIdx.y), the thread indices within each block (threadIdx.x, threadIdx.y), and the computed global indices (gx, gy) for completeness.
3. I chose a block dimension that is convenient for demonstration; 4x4 threads per block is a common choice. This gives 8*4 = 32 threads in x and 2*4 = 8 threads in y, totaling 256 threads overall. The specific block size is not critical to the requirement, but a small block size keeps the output manageable.
4. The kernel will use printf, which requires compiling with a compute capability of at least 2.0. The host code will simply launch the kernel, synchronize, and reset the device.
5. All necessary includes and error checking are provided. The program is self‑contained and can be compiled with `nvcc -o launch_2d launch_2d.cu` (or any name). 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints block, thread, and global 2D indices
__global__ void printIndices()
{
    // Compute global indices
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    // Print 2D indices
    printf("Block (%d,%d), Thread (%d,%d), Global (%d,%d)\n",
           blockIdx.x, blockIdx.y,
           threadIdx.x, threadIdx.y,
           gx, gy);
}

int main()
{
    // Define block and grid dimensions
    dim3 blockDim(4, 4);   // 4 threads in x, 4 in y
    dim3 gridDim(8, 2);    // 8 blocks in x, 2 in y

    // Launch the kernel
    printIndices<<<gridDim, blockDim>>>();

    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset device (optional)
    cudaDeviceReset();

    return 0;
}
```