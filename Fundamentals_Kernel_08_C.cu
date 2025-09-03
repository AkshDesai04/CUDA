/*
Aim of the program: Also print the grid's 2D dimensions (`gridDim.x`, `gridDim.y`).

[Thinking]
The task is to write a CUDA C program that demonstrates how to access and display the 2D dimensions of the kernel launch grid. CUDA allows device code to print directly to the console using `printf`, which is convenient for debugging and for this simple demonstration. The program will launch a kernel with a 2‑D grid configuration (for example, 4 blocks along the x‑dimension and 3 blocks along the y‑dimension) and a small block size. Inside the kernel we guard the `printf` call with an `if` that ensures only one thread (for instance, the first thread of the first block) performs the printing, so we don't get duplicate output from every thread.

After the kernel launch, the host code will synchronize the device and check for any launch errors, then exit. No additional libraries or host-side calculations are needed beyond the basic CUDA runtime API. The entire logic is self‑contained in a single .cu file, as requested.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel that prints the 2D grid dimensions */
__global__ void printGridDim()
{
    // Ensure only one thread does the printing to avoid duplicate output
    if (threadIdx.x == 0 && threadIdx.y == 0 &&
        blockIdx.x  == 0 && blockIdx.y  == 0)
    {
        printf("Grid dimensions: x = %d, y = %d\n", gridDim.x, gridDim.y);
    }
}

int main(void)
{
    /* Define a 2D grid and a small block */
    dim3 gridDim(4, 3);    // 4 blocks in x, 3 blocks in y
    dim3 blockDim(8, 8);   // 8x8 threads per block

    /* Launch the kernel */
    printGridDim<<<gridDim, blockDim>>>();

    /* Check for launch errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    /* Wait for GPU to finish */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Kernel launch completed successfully.\n");
    return 0;
}
