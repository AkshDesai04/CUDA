/*
Aim of the program: Also print the grid's dimension, `gridDim.x`, alongside the block index.

[Thinking]
- The program demonstrates how to access and display the grid dimension (`gridDim.x`) and the block index (`blockIdx.x`) from within a CUDA kernel.
- It uses a simple kernel that prints information only from one thread per block to avoid cluttering the output.
- Host code sets up a grid with a few blocks (e.g., 4) and a single thread per block, then launches the kernel.
- After the kernel execution, the host synchronizes the device and checks for errors.
- The output will show each block’s index and the total number of blocks in the grid.
- All necessary headers are included and the code is ready to be compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel that prints the block index and the grid dimension */
__global__ void printGridInfo()
{
    // Only the first thread in each block prints to keep output tidy
    if (threadIdx.x == 0)
    {
        printf("Block %d of %d\n", blockIdx.x, gridDim.x);
    }
}

int main()
{
    // Define block and grid dimensions
    const int threadsPerBlock = 1;   // One thread per block is enough for printing
    const int blocksPerGrid  = 4;    // Example grid with 4 blocks

    // Launch the kernel
    printGridInfo<<<blocksPerGrid, threadsPerBlock>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Synchronize to ensure all prints are complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset device (optional, cleans up)
    cudaDeviceReset();

    return 0;
}
