/*
Aim: Launch a kernel with a 2D grid of blocks, for example, 4x4 blocks, each with a single thread. Each thread should print its 2D block index (`blockIdx.x`, `blockIdx.y`).

Thinking process:

1. We need a CUDA kernel that runs on the GPU.  
2. Each thread in the grid is identified by `blockIdx`, `blockDim`, and `threadIdx`.  
3. The problem specifies a 2D grid of blocks (4x4) and only one thread per block.  
4. Inside the kernel, we will use `printf` (GPU printf) to output the block indices.  
5. In the host code, we define the grid dimensions as `dim3 grid(4, 4);` and block dimensions as `dim3 block(1, 1);`.  
6. After launching the kernel, we synchronize the device with `cudaDeviceSynchronize()` to make sure all prints are flushed before the program exits.  
7. The code includes necessary headers and error checking to keep it robust.  
8. The program is self‑contained, compiled with `nvcc` into a .cu file, and can be executed directly on a machine with an appropriate NVIDIA GPU.

Overall, the program demonstrates basic grid/block indexing and GPU-side printing, which is useful for debugging or understanding how CUDA maps work to the hardware.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the block indices
__global__ void printBlockIndices()
{
    // Each block has only one thread, but we still compute thread index
    printf("BlockIdx: (%d, %d)\n", blockIdx.x, blockIdx.y);
}

int main(void)
{
    // Define a 4x4 grid of blocks, 1 thread per block
    dim3 gridDim(4, 4);
    dim3 blockDim(1, 1);

    // Launch the kernel
    printBlockIndices<<<gridDim, blockDim>>>();

    // Check for kernel launch errors
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
        fprintf(stderr, "CUDA device synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset device before exiting (optional but clean)
    cudaDeviceReset();

    return 0;
}
