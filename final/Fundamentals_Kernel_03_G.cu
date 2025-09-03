/*
Launch with a 1D block of 32 threads. Have each thread print its warp ID (`threadIdx.x / 32`).

Thinking process:
- The program requirement specifies a 1D block consisting of 32 threads.
- Each thread must output its warp ID, computed as the integer division of its thread index by 32.
- Since there are only 32 threads, the division by 32 will result in 0 for all threads, but the code still follows the specification.
- Use CUDA device printf to print from within the kernel.
- Synchronize after kernel launch to ensure all prints are completed before the host program exits.
- Minimal error checking is added for robustness.
- The resulting file is a standalone CUDA source file (.cu) that can be compiled with nvcc and executed on a compatible GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printWarpID()
{
    // Compute warp ID as integer division by 32
    int warpID = threadIdx.x / 32;
    printf("Thread %d warp ID: %d\n", threadIdx.x, warpID);
}

int main(void)
{
    // Launch kernel with 1D block of 32 threads
    printWarpID<<<1, 32>>>();
    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
