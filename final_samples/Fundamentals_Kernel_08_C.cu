/*
Also print the grid's 2D dimensions (`gridDim.x`, `gridDim.y`).
This program demonstrates how to launch a CUDA kernel with a two‑dimensional grid
and print the grid dimensions from within the kernel using the device printf
facility. The kernel does not perform any heavy computation; its sole purpose
is to output the values of gridDim.x and gridDim.y. In the host code we
specify a 2D grid (for example, 4 blocks in the x‑dimension and 3 blocks in
the y‑dimension) and a 1×1 block. After launching the kernel we call
cudaDeviceSynchronize() to wait for the kernel to finish, ensuring that
the printed output appears before the program exits. Basic error checking is
included to catch launch failures or CUDA runtime errors. The program is
self‑contained and compiles with nvcc as a .cu file.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that prints the grid dimensions.
__global__ void printGridDims()
{
    // Using device printf to output grid dimensions.
    printf("Grid dimensions: x=%d, y=%d\n", gridDim.x, gridDim.y);
}

int main()
{
    // Define a 2D grid with 4 blocks in x and 3 blocks in y.
    dim3 gridDim(4, 3);
    // Define a 1×1 block.
    dim3 blockDim(1, 1);

    // Launch the kernel.
    printGridDims<<<gridDim, blockDim>>>();

    // Check for kernel launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the kernel to finish.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
