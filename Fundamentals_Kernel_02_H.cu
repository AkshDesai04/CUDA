/*
Print the values of `blockDim.x` and `gridDim.x` from within the kernel.
This program demonstrates how to access the built‑in CUDA variables
`blockDim.x` and `gridDim.x` inside a device kernel and print them
to the console using the device `printf`.  The kernel is launched with a
specified grid and block configuration; each thread prints the values
of these two dimension variables.  Since `printf` from the device is
asynchronous, a call to `cudaDeviceSynchronize()` is made on the host
after the kernel launch to ensure all output is flushed before the
program exits.  The program is kept intentionally simple: it
initializes the launch configuration, runs the kernel, and then exits.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printDimsKernel()
{
    // Each thread prints the block and grid dimensions.
    // Using %d to print integers.
    printf("Thread %d: blockDim.x = %d, gridDim.x = %d\n",
           threadIdx.x, blockDim.x, gridDim.x);
}

int main()
{
    // Define grid and block dimensions
    dim3 blockDim(4, 1, 1);  // 4 threads per block
    dim3 gridDim(2, 1, 1);   // 2 blocks in the grid

    // Launch the kernel
    printDimsKernel<<<gridDim, blockDim>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Reset the device (optional)
    cudaDeviceReset();

    return 0;
}
