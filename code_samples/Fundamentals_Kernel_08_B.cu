/*
Inside the kernel, calculate a flattened 1D block index from the 2D indices: `int id_1d = blockIdx.y * gridDim.x + blockIdx.x;`. Print this value.

Thinking:
- The goal is to illustrate how to compute a linear block index from 2‑D grid coordinates and to verify it via printing from the device.
- CUDA allows `printf` from kernels (device printf) provided the device supports it (compute capability ≥ 2.0). We will use it to output the computed `id_1d` for each block.
- To keep the example simple, we launch a single thread per block (blockDim.x = 1). Each block will execute the kernel and print its own flattened index.
- The grid will have multiple blocks arranged in a 2‑D grid (e.g., gridDim.x = 3, gridDim.y = 2), so we can observe several printed indices.
- After launching, we synchronize and check for errors on the host side.
- The code includes necessary headers, error checking, and a minimal main function to run the kernel.
- Compilation command (example): `nvcc -o block_index block_index.cu`
*/

#include <cstdio>
#include <cuda_runtime.h>

// Device kernel that prints the flattened block index
__global__ void printFlattenedBlockIndex()
{
    // Compute flattened 1D block index from 2D indices
    int id_1d = blockIdx.y * gridDim.x + blockIdx.x;

    // Use device printf to output the result
    printf("Block (%d, %d) -> Flattened ID: %d\n", blockIdx.x, blockIdx.y, id_1d);
}

int main()
{
    // Define grid dimensions (e.g., 3 blocks along x, 2 blocks along y)
    dim3 gridDim(3, 2);
    // Define block dimensions (1 thread per block)
    dim3 blockDim(1);

    // Launch the kernel
    printFlattenedBlockIndex<<<gridDim, blockDim>>>();

    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
