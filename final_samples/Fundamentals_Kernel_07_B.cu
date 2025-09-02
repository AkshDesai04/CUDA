/*
Aim: Inside the kernel, calculate a flattened 1D thread index from the 2D indices: `int id_1d = threadIdx.y * blockDim.x + threadIdx.x;`. Print this value.

Thinking:
- The program demonstrates how to compute a unique linear thread index when the kernel is launched with a 2D thread block configuration.
- The kernel will simply compute `id_1d` using the formula provided and use the CUDA device `printf` to output the value.
- The `printf` in device code requires the `-arch` flag and compilation with `nvcc`.  
- The main function will launch a single block of threads arranged in a 2D grid (e.g., 8x8) to keep the output manageable.
- After launching, the host will synchronize and check for errors.
- The code is self‑contained and can be compiled with:
  `nvcc -arch=sm_52 -o flattened_index flattened_index.cu` (or any other architecture).
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that computes a flattened 1D index from 2D thread indices and prints it
__global__ void print_flattened_index()
{
    // Compute flattened index: id_1d = threadIdx.y * blockDim.x + threadIdx.x
    int id_1d = threadIdx.y * blockDim.x + threadIdx.x;

    // Print the thread's 2D coordinates and the flattened index
    printf("Block (%d,%d), Thread (%d,%d) -> id_1d = %d\n",
           blockIdx.x, blockIdx.y,
           threadIdx.x, threadIdx.y,
           id_1d);
}

int main()
{
    // Define block and grid dimensions
    const dim3 threadsPerBlock(8, 8);  // 8x8 threads per block
    const dim3 numBlocks(1, 1);        // Single block for simplicity

    // Launch the kernel
    print_flattened_index<<<numBlocks, threadsPerBlock>>>();

    // Check for any launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
