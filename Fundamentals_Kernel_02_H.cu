```c
/*
Aim: Print the values of `blockDim.x` and `gridDim.x` from within the kernel.

Thought Process:
- The requirement is to create a simple CUDA C program that demonstrates how to access and print
  `blockDim.x` and `gridDim.x` inside a device kernel.
- To print from the device we will use `printf` (available in device code since CUDA 2.0).
- We need a kernel that will be launched with some number of blocks and threads; each thread can
  print the values, but to avoid flooding the output we’ll let only the first thread of each block
  print the values.
- In `main` we will define grid and block dimensions, launch the kernel, then synchronize
  and check for errors.
- The program will be compiled with `nvcc` and will run on a CUDA-capable GPU.

Implementation Steps:
1. Include `<stdio.h>` and `<cuda_runtime.h>`.
2. Define a kernel `printDimsKernel` that uses `blockDim.x` and `gridDim.x`.
3. In `main`, set `dim3 block(4); dim3 grid(2);`.
4. Launch the kernel: `printDimsKernel<<<grid, block>>>( );`.
5. Synchronize and handle any launch or runtime errors.
6. Provide a simple exit status.

The final file below is a complete `.cu` source file that can be compiled and run with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printDimsKernel()
{
    // Let only the first thread of each block print to avoid redundant output
    if (threadIdx.x == 0)
    {
        printf("Block %d: blockDim.x = %d, gridDim.x = %d\n",
               blockIdx.x, blockDim.x, gridDim.x);
    }
}

int main(void)
{
    // Define grid and block dimensions
    dim3 block(4);  // 4 threads per block
    dim3 grid(2);   // 2 blocks in the grid

    // Launch the kernel
    printDimsKernel<<<grid, block>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset the device (optional but good practice)
    cudaDeviceReset();

    return 0;
}
```