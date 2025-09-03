/*
Aim of the program:
Change the launch configuration to a non-square 2D block, like 16x4 threads, and print the
2D indices.

Thinking:
To demonstrate a non-square 2D block configuration in CUDA, I chose a block size of 16
threads in the x-dimension and 4 threads in the y-dimension. This creates a rectangular block
with 64 threads per block. The kernel will use printf to output each thread's local
( threadIdx.x, threadIdx.y ) and global ( blockIdx.x, blockIdx.y ) coordinates as well as the
computed linear global indices (globalX, globalY). A small grid of 2x2 blocks is used for
clarity, resulting in a total of 128 threads. After launching the kernel, the host code waits
for completion with cudaDeviceSynchronize() before exiting. The program is self‑contained
and can be compiled with nvcc as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printIndicesKernel()
{
    // Compute 2D local indices
    int localX = threadIdx.x;
    int localY = threadIdx.y;

    // Compute 2D global indices
    int globalX = blockIdx.x * blockDim.x + localX;
    int globalY = blockIdx.y * blockDim.y + localY;

    // Print the indices
    printf("Thread (%d,%d) in block (%d,%d) -> global (%d,%d)\n",
           localX, localY, blockIdx.x, blockIdx.y, globalX, globalY);
}

int main()
{
    // Define a non-square 2D block (16x4 threads)
    dim3 blockDim(16, 4);  // 16 threads in x, 4 threads in y

    // Define a small 2D grid (2x2 blocks)
    dim3 gridDim(2, 2);    // 2 blocks in x, 2 blocks in y

    // Launch the kernel
    printIndicesKernel<<<gridDim, blockDim>>>();

    // Wait for the kernel to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
