```cuda
/*
Change the launch configuration to a non-square 2D grid, like 8x2 blocks, and print the 2D indices.

Thinking:
The goal is to demonstrate a CUDA kernel that uses a two‑dimensional grid layout that is not square.
We will configure the launch as dim3 grid(8, 2) to get 8 blocks in the x‑direction and 2 blocks in the y‑direction.
For each block we will use a two‑dimensional blockDim, e.g. dim3 block(8, 8), which gives 64 threads per block.
Inside the kernel we will print the thread and block indices in both x and y directions so that
the output shows the 2D nature of the launch configuration.
We will use printf from within the kernel (available on devices with compute capability >= 2.0).
Error checking will be done after the kernel launch, and the host will synchronize and report any errors.

The program is self‑contained, compiled as a .cu file.  It prints a line for each thread in the form:
  "Thread (tx,ty) in block (bx,by)"
which clearly shows the 2D indices of both the thread and the block.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_indices()
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    printf("Thread (%d,%d) in block (%d,%d)\n", tx, ty, bx, by);
}

int main()
{
    // Define a non-square 2D grid: 8 blocks in x, 2 blocks in y
    dim3 gridDim(8, 2);
    // Define a 2D block: 8 threads in x, 8 threads in y
    dim3 blockDim(8, 8);

    // Launch kernel
    print_indices<<<gridDim, blockDim>>>();

    // Synchronize and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset device for cleanliness
    cudaDeviceReset();
    return 0;
}
```