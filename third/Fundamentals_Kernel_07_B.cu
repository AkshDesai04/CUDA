/*
Aim of the program:
Inside the kernel, calculate a flattened 1D thread index from the 2D indices:
int id_1d = threadIdx.y * blockDim.x + threadIdx.x;
Print this value.

Thinking:
The goal is to illustrate how to derive a linear index from the two-dimensional
thread coordinates provided by CUDA.  Each thread has a 2‑D index
(threadIdx.x, threadIdx.y).  By using the block dimensions we can map these
coordinates to a single integer:
    id_1d = threadIdx.y * blockDim.x + threadIdx.x
This is equivalent to row-major ordering within a block.

To demonstrate this we:
1. Write a __global__ kernel that computes id_1d for each thread and prints
   the thread coordinates along with the resulting 1D index.  Device printf
   is used, which requires including <stdio.h> and compiling with a GPU
   architecture that supports it.
2. In main() we configure a simple 2‑D grid of blocks and a 2‑D block of
   threads, launch the kernel, and synchronize.
3. The program prints each thread’s flattened index, which can be verified
   by eye or by comparing against the expected sequence.

The code is self‑contained, minimal, and can be compiled with nvcc:
    nvcc -arch=sm_20 -o flattened_index flattened_index.cu
*/

#include <stdio.h>

__global__ void computeFlattenedIndex()
{
    int id_1d = threadIdx.y * blockDim.x + threadIdx.x;
    printf("Thread (%3d, %3d) -> flattened index %3d\n",
           threadIdx.x, threadIdx.y, id_1d);
}

int main(void)
{
    dim3 blockDim(8, 8);   // 8x8 threads per block
    dim3 gridDim(2, 2);    // 2x2 blocks

    computeFlattenedIndex<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();

    return 0;
}
