```cuda
/*
Aim: Combine with 2D blocks. Launch a 2x2 grid of 4x4 blocks. Have each thread calculate and print its global 2D index: 
`int global_x = blockIdx.x * blockDim.x + threadIdx.x;` and `int global_y = ...`.

Thought process:
1. We need a kernel that runs with a 2D grid and 2D blocks. 
2. Grid dimensions: 2x2 (so `dim3 grid(2,2)`). 
3. Block dimensions: 4x4 (so `dim3 block(4,4)`).
4. Each thread calculates its global coordinates using the formulas:
   ```
   int global_x = blockIdx.x * blockDim.x + threadIdx.x;
   int global_y = blockIdx.y * blockDim.y + threadIdx.y;
   ```
5. We use `printf` inside the kernel to output the global indices. Device-side `printf` requires `-arch=sm_20` or higher and is available in CUDA 2.0+.
6. After launching the kernel, we call `cudaDeviceSynchronize()` to ensure all output is flushed before the program exits.
7. Include minimal error checking for the kernel launch.

The resulting file is a complete, self‑contained CUDA C program that can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the global 2D index of each thread
__global__ void print_global_indices()
{
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    printf("Thread (bx=%d, by=%d, tx=%d, ty=%d) -> Global (x=%d, y=%d)\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
           global_x, global_y);
}

int main()
{
    // Define 2D grid and block dimensions
    dim3 blockDim(4, 4);   // 4x4 threads per block
    dim3 gridDim(2, 2);    // 2x2 blocks

    // Launch kernel
    print_global_indices<<<gridDim, blockDim>>>();

    // Synchronize to ensure kernel completes and output is flushed
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
```