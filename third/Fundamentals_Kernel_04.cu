```cuda
/*
Aim of the program: Write a kernel that is launched with a 1D grid of 8 blocks, where each block has only one thread. Each thread should print its own block index, which is accessible via `blockIdx.x`.

Thinking:
1. We need to write a CUDA kernel that will be executed on the device.
2. Each thread will print its block index using the built‑in `printf` function, which is available for device code on compute capability >= 2.0.
3. The launch configuration requires 8 blocks in a 1D grid, and each block should contain only a single thread.
4. Therefore, the kernel will be launched with `<<<8, 1>>>`.
5. After launching the kernel, we must synchronize the device to ensure all prints are flushed before the host program exits.
6. Include `<stdio.h>` for the printf function and `<cuda_runtime.h>` for the CUDA runtime API.
7. The host program will call `cudaDeviceReset()` to cleanly shut down the device before returning.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printBlockIdx()
{
    // Each thread prints its block index
    printf("Block %d\n", blockIdx.x);
}

int main()
{
    // Launch kernel with 8 blocks and 1 thread per block
    printBlockIdx<<<8, 1>>>();

    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Clean up and reset device
    cudaDeviceReset();
    return 0;
}
```