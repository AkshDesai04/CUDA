```cpp
/*
Aim of the program:
Declare dim3 grid(2, 2); and dim3 block(8, 8). Launch a kernel and print the x and y components of gridDim and blockDim.

Thinking:
1. Use the built‑in variables gridDim and blockDim in device code to obtain the dimensions of the grid and block that launched the kernel.
2. Since we only need to print these values once, we restrict the printing to thread (0,0) of each block to avoid duplicate output.
3. Device printf is available on recent CUDA-capable GPUs, so we can use printf directly from the kernel.
4. The kernel will be launched with the specified grid and block dimensions.
5. After the kernel launch, we synchronize the device to ensure all output has been flushed before the program exits.
6. The code is written in standard C++ with CUDA extensions and is fully compilable with nvcc.

*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printDims()
{
    // Only one thread per block will print to avoid duplicate output
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("gridDim: (%d, %d)\n", gridDim.x, gridDim.y);
        printf("blockDim: (%d, %d)\n", blockDim.x, blockDim.y);
    }
}

int main()
{
    // Declare grid and block dimensions
    dim3 grid(2, 2);
    dim3 block(8, 8);

    // Launch kernel
    printDims<<<grid, block>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    return 0;
}
```