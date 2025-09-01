```c
/*
Aim of the program: Also print the grid's 2D dimensions (`gridDim.x`, `gridDim.y`).

Thinking:
- The objective is to demonstrate how a CUDA kernel can access and display the 2D grid dimensions.
- We'll write a simple kernel that prints the thread indices, block indices, and the grid dimensions.
- `gridDim` is a built‑in variable available inside a kernel, so we can use it directly in `printf`.
- The host will launch the kernel with a 2D grid and a 2D block configuration.
- After the kernel execution, the host will also print the grid dimensions to confirm consistency.
- Device side `printf` requires CUDA 2.0 or later; compilation with `nvcc` and an appropriate compute capability
  will allow the output to be visible.
- Basic error checking is added to catch any launch or synchronization failures.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printGridInfo()
{
    // Each thread prints its own identifiers along with the grid dimensions
    printf("Thread (%d,%d) in Block (%d,%d) of Grid (%d,%d)\n",
           threadIdx.x, threadIdx.y,
           blockIdx.x, blockIdx.y,
           gridDim.x, gridDim.y);
}

int main()
{
    // Define a 2D block and a 2D grid
    dim3 blockDim(4, 4); // 16 threads per block
    dim3 gridDim(2, 3);  // 6 blocks in total

    // Launch the kernel
    printGridInfo<<<gridDim, blockDim>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for the GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA synchronize error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Print the grid dimensions from the host as well
    printf("Host: Grid dimensions are (%d, %d)\n", gridDim.x, gridDim.y);

    return 0;
}
```