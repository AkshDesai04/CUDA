/*
Launch the kernel with a 2D block of (8, 8) threads and have each thread print only its `threadIdx.x`.

My approach:
- The program will be a minimal CUDA C program.
- A kernel `printThreadX` will run on the device and call `printf("%d\n", threadIdx.x);` for each thread.
- The host will launch this kernel with a block dimension of (8,8) and a single grid block.
- After launching, `cudaDeviceSynchronize()` ensures that the device completes before the program exits.
- No additional error checking or memory allocations are needed for this simple example.
- The output will show 64 lines (one per thread in the 8×8 block), each containing the `threadIdx.x` value of that thread.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printThreadX()
{
    // Each thread prints its own threadIdx.x
    printf("%d\n", threadIdx.x);
}

int main()
{
    // Define block and grid dimensions
    dim3 blockDim(8, 8);
    dim3 gridDim(1, 1);

    // Launch kernel
    printThreadX<<<gridDim, blockDim>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}
