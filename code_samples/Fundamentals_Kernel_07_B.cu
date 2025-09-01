/*
 Inside the kernel, calculate a flattened 1D thread index from the 2D indices: `int id_1d = threadIdx.y * blockDim.x + threadIdx.x;`. Print this value.
 
 The goal of this program is to demonstrate how to compute a single linear index from 2‑D thread coordinates in a CUDA kernel and to output that index for each thread. 
 We choose a simple block and grid configuration (2×2 threads per block and 2×2 blocks in the grid) so that there are 16 threads in total. 
 Each thread will compute its own flattened index and print it using `printf`. Device side printf requires a compute capability of at least 2.0 and must be enabled by compiling with the appropriate flags (which the user must supply). 
 The host code launches the kernel, synchronizes the device, and checks for errors. 
 All standard CUDA error handling is included for robustness. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void flatten_and_print()
{
    // Compute a flattened 1D index from 2D thread indices
    int id_1d = threadIdx.y * blockDim.x + threadIdx.x;
    // Print thread coordinates and flattened index
    printf("Thread (%d, %d) flattened id: %d\n", threadIdx.x, threadIdx.y, id_1d);
}

int main()
{
    // Define block and grid dimensions
    dim3 threadsPerBlock(2, 2); // 2x2 threads per block
    dim3 numBlocks(2, 2);       // 2x2 blocks in grid

    // Launch the kernel
    flatten_and_print<<<numBlocks, threadsPerBlock>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Reset the device (optional)
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
