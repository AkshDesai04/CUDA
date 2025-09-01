/*
Declare `dim3 grid(2, 2);` and `dim3 block(8, 8);`. Launch a kernel and print the `x` and `y` components of `gridDim` and `blockDim`.

Thinking:
1. The goal is to demonstrate how to access the grid and block dimensions inside a CUDA kernel.
2. We'll set up the grid and block using `dim3 grid(2, 2);` and `dim3 block(8, 8);` as requested.
3. Inside the kernel we can simply print the values of `gridDim.x`, `gridDim.y`, `blockDim.x`, and `blockDim.y`.
4. To avoid cluttering the output with repeated prints from every thread, we restrict the printing to the first thread of each block (`threadIdx.x == 0 && threadIdx.y == 0`).
5. The kernel will be launched with the specified grid and block configuration.
6. After launching, we call `cudaDeviceSynchronize()` to ensure all kernel output is flushed before the program exits.
7. The program includes minimal error checking for the kernel launch and device synchronization.
8. The output will show the requested `x` and `y` components of both `gridDim` and `blockDim`.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that prints the grid and block dimensions
__global__ void printDims()
{
    // Only one thread per block prints to avoid duplicate output
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        printf("gridDim.x = %d, gridDim.y = %d\n", gridDim.x, gridDim.y);
        printf("blockDim.x = %d, blockDim.y = %d\n", blockDim.x, blockDim.y);
    }
}

int main()
{
    // Define grid and block dimensions
    dim3 grid(2, 2);
    dim3 block(8, 8);

    // Launch the kernel
    printDims<<<grid, block>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d, line(%d)\n", err, __LINE__);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
