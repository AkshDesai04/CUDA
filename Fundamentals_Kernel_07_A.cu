/*
Aim of the program: Change the launch configuration to a non-square 2D block, like 16x4 threads, and print the 2D indices.

Thinking:
1. The original program likely used a square block of threads (e.g., 16x16). We want to switch to a non‑square block to demonstrate flexibility of launch configuration.
2. A 16x4 block provides 64 threads per block. The kernel will use `threadIdx.x` and `threadIdx.y` to access the 2‑D thread indices.
3. For simplicity and to keep the example self‑contained, the kernel takes a dummy device pointer parameter. We allocate a single integer on the device in `main` and pass its pointer; the kernel doesn't use it.
4. We set up a 2×2 grid of blocks so that the total number of threads is manageable for output (2×2×16×4 = 256 threads). Each thread prints its block and thread coordinates using `printf`. CUDA's `printf` inside kernels writes to stdout asynchronously; therefore we call `cudaDeviceSynchronize()` after the kernel launch to ensure all output is flushed before the program exits.
5. Basic error checking is added via `cudaGetLastError()` after the kernel launch.
6. The code includes necessary headers (`<stdio.h>`, `<cuda_runtime.h>`), defines the kernel, sets launch parameters, and cleans up device memory.

Compile with:
    nvcc -arch=sm_61 non_square_block.cu -o non_square_block

Run:
    ./non_square_block
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printIndicesKernel(int *dummy)
{
    // Print the 2D indices of each thread within its block
    printf("Block (%d,%d), Thread (%d,%d)\n",
           blockIdx.x, blockIdx.y,
           threadIdx.x, threadIdx.y);
}

int main(void)
{
    // Allocate dummy memory on the device
    int *d_dummy;
    cudaError_t err = cudaMalloc((void**)&d_dummy, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Define a non-square 2D block configuration: 16x4 threads per block
    const int blockDimX = 16;
    const int blockDimY = 4;
    dim3 threadsPerBlock(blockDimX, blockDimY);

    // Define grid dimensions (2x2 blocks)
    const int gridDimX = 2;
    const int gridDimY = 2;
    dim3 numBlocks(gridDimX, gridDimY);

    // Launch the kernel
    printIndicesKernel<<<numBlocks, threadsPerBlock>>>(d_dummy);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_dummy);
        return 1;
    }

    // Wait for the kernel to finish and ensure all prints are flushed
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Device synchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_dummy);
        return 1;
    }

    // Clean up
    cudaFree(d_dummy);
    return 0;
}
