```cu
/*
Aim of the program:
Change the launch configuration to a non-square 2D grid, like 8x2 blocks, and print the 2D indices.

Thinking process:
- To illustrate a non-square grid launch configuration, I chose a grid of 8 blocks in the x-dimension and 2 blocks in the y-dimension.
- Each block will have a simple 4x4 thread configuration so that the total number of threads (8*2*4*4 = 256) is manageable for printing.
- The kernel prints both the block and thread 2D indices, as well as the computed global thread indices for clarity.
- Using `printf` inside the kernel requires the device to be compiled with `-arch=sm_20` or higher; NVCC handles this automatically.
- After launching the kernel, the host calls `cudaDeviceSynchronize()` to ensure all device-side prints are flushed before the program exits.
- Error checking is minimal to keep the example focused on the launch configuration and index printing.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printIndices()
{
    // Capture block and thread indices
    int gx = blockIdx.x;
    int gy = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockDim.x;
    int by = blockDim.y;

    // Compute global thread indices
    int gx_global = gx * bx + tx;
    int gy_global = gy * by + ty;

    // Print the 2D indices
    printf("Block (%2d,%2d), Thread (%2d,%2d) -> Global (%2d,%2d)\n",
           gx, gy, tx, ty, gx_global, gy_global);
}

int main()
{
    // Define grid and block dimensions
    dim3 gridDim(8, 2);   // Non-square grid: 8 blocks in x, 2 in y
    dim3 blockDim(4, 4);  // 4x4 threads per block

    // Launch kernel
    printIndices<<<gridDim, blockDim>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
```