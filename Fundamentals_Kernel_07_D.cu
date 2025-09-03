/*
Launch a 3D block of (4, 4, 4) threads. Have each thread print its 3D thread index (`threadIdx.x`, `threadIdx.y`, `threadIdx.z`).

Thinking:
1. The task requires a single block with dimensions (4,4,4), so the blockDim will be set to 4 in each of the x, y, and z dimensions.
2. We need a kernel that accesses `threadIdx.x`, `threadIdx.y`, and `threadIdx.z` and prints them. CUDA kernels can use `printf` for output, but the output will only appear after `cudaDeviceSynchronize()` on the host side.
3. No additional grid dimensions are necessary; a single block (grid(1,1,1)) is sufficient for the example.
4. Basic error checking can be omitted for brevity, but we will still call `cudaDeviceSynchronize()` to ensure that all kernel prints are flushed before the program exits.
5. The code will be a minimal CUDA C program that can be compiled with `nvcc`. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printThreadIdx()
{
    printf("ThreadIdx: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
    dim3 block(4, 4, 4);
    dim3 grid(1, 1, 1);

    printThreadIdx<<<grid, block>>>();
    cudaDeviceSynchronize();

    return 0;
}
