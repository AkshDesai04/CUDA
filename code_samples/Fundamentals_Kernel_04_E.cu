```cpp
/*
Aim of the program:
Launch the kernel with a 2D grid of (4, 4) blocks (each with one thread) and have each thread print only its blockIdx.x.

Thinking process:
- Use CUDA C/C++ to write a simple kernel that prints the blockIdx.x value from each thread.
- Since each block contains only one thread, the thread index will always be (0,0).
- The kernel will use the standard device-side printf to output the value.
- On the host side, configure a 2D grid with dimensions dim3 grid(4,4) and a block dimension dim3 block(1,1).
- After launching the kernel, synchronize the device to ensure all prints complete before the program exits.
- Add basic error checking after kernel launch and device sync for robustness.
- The code is self-contained and can be compiled with nvcc: `nvcc -arch=sm_XX -o print_blockx print_blockx.cu`.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that prints the blockIdx.x of each thread
__global__ void printBlockIdxX()
{
    // Each block has only one thread
    printf("%d\n", blockIdx.x);
}

int main()
{
    // Define a 2D grid of (4,4) blocks
    dim3 gridDim(4, 4);
    // Each block contains one thread
    dim3 blockDim(1, 1);

    // Launch kernel
    printBlockIdxX<<<gridDim, blockDim>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for the GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA sync failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
```