/*
Aim of the program: Also print the block's dimension, blockDim.x, alongside the thread index.

My thinking:
The user wants a simple CUDA program that demonstrates how to access and print the block dimension (blockDim.x) and the thread index within each thread. To keep things clear, I will:

1. Define a kernel that uses `printf` to output the thread's global index (`blockIdx.x * blockDim.x + threadIdx.x`), the thread's local index (`threadIdx.x`), and the block dimension (`blockDim.x`).
2. In `main`, launch a single block with a few threads (e.g., 8 threads) so that the output is easy to read.
3. Use standard CUDA error checking to ensure the kernel launch succeeded.
4. Add comments in the code to explain each part.
5. Ensure the file is a proper `.cu` source file that can be compiled with `nvcc`.

The code below implements this plan. It will compile with `nvcc` and, when run, print a line per thread showing the requested information.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the thread index and block dimension
__global__ void printThreadInfo()
{
    // Compute global thread index
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // Print the thread's local index, global index, and block dimension
    printf("Thread %d (global %d) in block %d: blockDim.x = %d\n",
           threadIdx.x, globalIdx, blockIdx.x, blockDim.x);
}

int main()
{
    // Define grid and block dimensions
    dim3 blockDim(8);   // 8 threads per block
    dim3 gridDim(1);    // 1 block

    // Launch the kernel
    printThreadInfo<<<gridDim, blockDim>>>();
    // Synchronize to ensure all prints complete
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    // Optional: reset the device (clean up)
    cudaDeviceReset();

    return 0;
}
