/*
Aim: Also print the grid's 2D dimensions (`gridDim.x`, `gridDim.y`).

Thinking:
- The user requests a CUDA program that prints the grid's 2D dimensions. 
- We will write a minimal CUDA .cu file that launches a kernel with a 2D grid configuration and a 2D block configuration.
- Inside the kernel we will use `printf` to output the grid dimensions (`gridDim.x`, `gridDim.y`) and block dimensions for clarity.
- To avoid excessive output, only thread (0,0,0) will perform the printing.
- We will include basic error checking for the kernel launch.
- The host code will configure a simple 2D grid (e.g., 4x3) and a 2D block (e.g., 8x8), launch the kernel, synchronize, and exit.
- This satisfies the requirement to print the grid's 2D dimensions and provides a self‑contained CUDA example that compiles into a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printGridInfo()
{
    // Only thread (0,0,0) prints the information to avoid clutter
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        printf("Grid dimensions: x=%d, y=%d, z=%d\n",
               gridDim.x, gridDim.y, gridDim.z);
        printf("Block dimensions: x=%d, y=%d, z=%d\n",
               blockDim.x, blockDim.y, blockDim.z);
        printf("Block index: x=%d, y=%d, z=%d\n",
               blockIdx.x, blockIdx.y, blockIdx.z);
        printf("Thread index: x=%d, y=%d, z=%d\n",
               threadIdx.x, threadIdx.y, threadIdx.z);
    }
}

int main()
{
    // Define a 2D grid: 4 blocks in x, 3 blocks in y
    dim3 gridDim(4, 3, 1);
    // Define a 2D block: 8 threads in x, 8 threads in y
    dim3 blockDim(8, 8, 1);

    // Launch the kernel
    printGridInfo<<<gridDim, blockDim>>>();

    // Synchronize to ensure kernel has finished before printing from host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Reset device (optional, good practice for some test environments)
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
