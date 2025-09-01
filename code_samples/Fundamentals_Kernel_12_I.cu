/*
Initialize a `dim3` variable with all three dimensions, e.g., `dim3 grid(4, 2, 2);`. 
[Thinking]
We want to write a minimal CUDA program that shows the initialization of a dim3 variable
with all three dimensions. The dim3 type is a CUDA built‑in type used to describe
grid or block dimensions. In this program we create a host variable
`grid` with the three components (4,2,2).  We then launch a very simple kernel
using that grid and a small block.  The kernel prints some information using
`printf` so that when we run the program we can see that the launch dimensions
were applied correctly.  The host prints the values of the grid and block
dimensions before launching.  Finally we call `cudaDeviceSynchronize()` to
ensure the kernel has finished and check for any launch errors.  The program
is fully self‑contained and compiles with nvcc.
*/

#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel()
{
    printf("Block (%d, %d, %d), Thread (%d, %d, %d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
    dim3 grid(4, 2, 2);
    dim3 block(8, 8, 1);

    std::cout << "Launching kernel with grid dimensions: (" << grid.x << ", " << grid.y << ", " << grid.z << ")\n";
    std::cout << "Block dimensions: (" << block.x << ", " << block.y << ", " << block.z << ")\n";

    kernel<<<grid, block>>>();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    return 0;
}
