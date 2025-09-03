/*
Aim: Launch the printing kernel with 10 blocks of one thread. Observe the output.

Thinking:
The assignment requires a simple CUDA program that demonstrates launching a kernel
with 10 blocks, each containing a single thread, and printing a message from
each thread.  The kernel will use the built‑in `printf` function, which is
available in device code when compiled with NVCC on GPUs that support it
(typically compute capability 2.0 and above).  The output should therefore
display ten lines, one from each block.

Implementation details:
1. Include the standard headers `<stdio.h>` and `<cuda_runtime.h>`.
2. Define a global kernel `printKernel` that calls `printf` with the block and
   thread indices.
3. In `main`, launch the kernel with a grid of 10 blocks and a block size of
   1 thread: `printKernel<<<10, 1>>>();`.
4. After the launch, call `cudaDeviceSynchronize()` to ensure the kernel has
   finished before the program exits.
5. Perform basic error checking after the launch and synchronization to
   catch any runtime errors.
6. Return 0 from `main`.

The program is written in standard C, but with CUDA extensions, and should be
compiled with `nvcc`.  The entire source is contained in a single `.cu` file
as requested.*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printKernel()
{
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main(void)
{
    // Launch kernel: 10 blocks, 1 thread per block
    printKernel<<<10, 1>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
