/*
Launch the kernel with a 2D grid of (4, 4) blocks (each with one thread) and have each thread print only its `blockIdx.x`.

Thought process:
- The user wants a simple CUDA program demonstrating kernel launch configuration.
- We need a 2D grid of 4x4 blocks. Each block has only 1 thread. So we use dim3 grid(4,4) and dim3 block(1).
- The kernel will simply print blockIdx.x. Because each block only has one thread, threadIdx.x, y, z will all be 0. We can ignore them.
- Use device-side printf. On recent CUDA versions, printf is supported on compute capability >=2.0. We'll assume this is fine.
- After launching the kernel, we need to synchronize to make sure prints complete.
- The code will include main, kernel, grid/block config, and a cleanup. We'll use cudaDeviceReset() for cleanup.

Edge cases: 
- Ensure we check for errors after kernel launch? Optional. For brevity, we might skip error checking, but we can include a simple check.
- Provide comments to explain each part.

We'll write the code accordingly.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the blockIdx.x of each thread
__global__ void printBlockIdxX()
{
    // Each block has only one thread, so threadIdx is always 0
    printf("Thread in block (%d, %d) has blockIdx.x = %d\n",
           blockIdx.x, blockIdx.y, blockIdx.x);
}

int main()
{
    // Define a 2D grid of 4x4 blocks, each block with 1 thread
    dim3 gridDim(4, 4);
    dim3 blockDim(1, 1, 1);

    // Launch the kernel
    printBlockIdxX<<<gridDim, blockDim>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Reset the device and exit
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
