```cuda
/*
Aim of the program: Modify the kernel so that blocks with `blockIdx.x == blockIdx.y` (on the diagonal) print a special message.

Thought process:
1. We need a CUDA kernel that runs on a grid of blocks. Each block will have some number of threads, but for the purpose of printing a message per block, we can use a single thread per block (e.g., 1x1 thread block).
2. Inside the kernel, each thread (there is only one per block) will check whether the block coordinates satisfy blockIdx.x == blockIdx.y. If so, we call `printf` with a special message that includes the block indices. Otherwise, we print a normal message for non‑diagonal blocks.
3. Because `printf` is a device function, we must compile with a compute capability that supports device printf (≥3.0). We'll use `-arch=sm_30` or higher.
4. The host code will set up a 2D grid, e.g., 4x4 blocks, launch the kernel, and synchronize, checking for errors.
5. The code will include necessary headers (`stdio.h` and `cuda_runtime.h`), define the kernel, and in `main()` launch the kernel and handle errors.

This results in a self‑contained .cu file that, when compiled with nvcc, will print messages indicating which blocks are on the diagonal and which are not.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints a special message for diagonal blocks
__global__ void diagKernel()
{
    // Each block is responsible for its own message
    if (blockIdx.x == blockIdx.y)
    {
        printf("Diagonal block [%d, %d] - special message\n",
               blockIdx.x, blockIdx.y);
    }
    else
    {
        printf("Block [%d, %d]\n", blockIdx.x, blockIdx.y);
    }
}

int main()
{
    // Define grid and block dimensions
    dim3 threadsPerBlock(1, 1, 1);      // 1 thread per block
    dim3 numBlocks(4, 4, 1);            // 4x4 grid of blocks

    // Launch kernel
    diagKernel<<<numBlocks, threadsPerBlock>>>();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
```