```cpp
/*
Aim of the program (word to word from the prompt):
Inside the kernel, calculate a flattened 1D block index from the 2D indices:
int id_1d = blockIdx.y * gridDim.x + blockIdx.x;
Print this value.

Thought process:
- The goal is to demonstrate how a 2‑D grid of thread blocks can be flattened into a single dimension.
- In CUDA, each block has a 2‑D index (blockIdx.x, blockIdx.y) and the grid itself also has a 2‑D dimension (gridDim.x, gridDim.y).
- The formula `id_1d = blockIdx.y * gridDim.x + blockIdx.x` maps the 2‑D block coordinates to a linear index by first counting all blocks in previous rows (blockIdx.y * gridDim.x) and then adding the column offset (blockIdx.x).
- To observe this mapping, we will launch a kernel with a 2‑D grid. Each block will launch a single thread that prints its 2‑D index and the computed 1‑D index.
- `printf` can be used inside a kernel to output text to the console (requires a GPU architecture of compute capability 2.0 or higher and that the device supports device-side printf).
- The host code will set up a 2‑D grid (e.g., 4 columns × 3 rows), launch the kernel with one thread per block, synchronize, and then exit.
- Error checking will be minimal but sufficient to catch launch failures.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the 2D block index and its flattened 1D counterpart
__global__ void printBlockIndex()
{
    // Compute flattened 1D block index
    int id_1d = blockIdx.y * gridDim.x + blockIdx.x;

    // Print the result. The thread count is 1 per block, so this runs once per block.
    printf("Block (x=%d, y=%d) -> 1D id = %d\n",
           blockIdx.x, blockIdx.y, id_1d);
}

int main(void)
{
    // Define grid dimensions: 4 columns (x) × 3 rows (y)
    dim3 gridDim(4, 3);
    // Use a single thread per block
    dim3 blockDim(1);

    // Launch the kernel
    printBlockIndex<<<gridDim, blockDim>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```