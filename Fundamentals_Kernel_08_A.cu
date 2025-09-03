/*
Change the launch configuration to a non-square 2D grid, like 8x2 blocks, and print the 2D indices.
In this program we will create a CUDA kernel that prints the 2‑D indices of each block and each thread within that block.  The launch configuration is deliberately chosen to be non‑square: 8 blocks along the X‑axis and 2 blocks along the Y‑axis.  For each block we will also use a small 2‑D thread block of size 4×4 so that the program produces a manageable amount of output while still demonstrating 2‑D indexing.  We will use the device side `printf` which is available on compute capability 2.0 and higher.  After launching the kernel we call `cudaDeviceSynchronize()` to ensure that all device printf statements are flushed to the host console before the program exits.  The host code is minimal – it just sets up the grid and block dimensions, launches the kernel, checks for errors, and synchronizes.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints block and thread 2D indices
__global__ void printIndicesKernel()
{
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    // Print the indices from each thread
    printf("Block (%d,%d), Thread (%d,%d)\n", blockX, blockY, threadX, threadY);
}

int main(void)
{
    // Define block and grid dimensions
    dim3 threadsPerBlock(4, 4); // 4 threads in X, 4 in Y
    dim3 numBlocks(8, 2);       // 8 blocks in X, 2 in Y

    // Launch the kernel
    printIndicesKernel<<<numBlocks, threadsPerBlock>>>();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
