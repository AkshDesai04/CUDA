/*
Initialize a `dim3` variable with all three dimensions, e.g., `dim3 grid(4, 2, 2);`.

Thinking:
- The goal is to demonstrate how to create a `dim3` object with all three dimensions explicitly set.
- `dim3` is a simple struct used by CUDA to specify block and grid sizes.
- To showcase its usage, the program will:
  1. Instantiate a `dim3` grid with the values (4, 2, 2) as specified.
  2. Instantiate a block dimension for completeness.
  3. Launch a trivial kernel that prints the grid and block dimensions it sees.
  4. Print the same dimensions on the host side for confirmation.
- The program uses `printf` from the device to output from the kernel, which requires the device to support `printf`. 
- The host code performs minimal error checking and synchronizes the device after kernel launch.
- This minimal example is self‑contained and can be compiled with `nvcc`.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void dummyKernel()
{
    printf("Kernel launch info:\n");
    printf("  Grid dimensions (host view): (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
    printf("  Block dimensions (host view): (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
}

int main()
{
    // Initialize dim3 with all three dimensions
    dim3 grid(4, 2, 2);      // grid of size 4x2x2
    dim3 block(8, 8, 8);     // block of size 8x8x8

    // Print host-side dimensions
    printf("Host grid dimensions: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    printf("Host block dimensions: (%d, %d, %d)\n", block.x, block.y, block.z);

    // Launch the dummy kernel
    dummyKernel<<<grid, block>>>();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
