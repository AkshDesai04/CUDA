/*
Explain in a comment why this formula for global ID works.
I started by recalling how CUDA organizes threads into a hierarchy of blocks and grids. Each block has a unique index blockIdx and each thread within a block has a unique index threadIdx. The total number of threads in the grid is blockDim.x * gridDim.x (assuming 1D launch). To map a 2D or 3D index space into a flat linear index (global ID), we multiply the block index by the number of threads per block and then add the intra-block thread index. This works because:
- blockIdx.x identifies which block in the grid we are in. Multiplying by blockDim.x gives us the starting index of that block in the global sequence.
- threadIdx.x is the offset within the block.
- Adding the two yields a unique global index for every thread in the grid, ranging from 0 to (gridDim.x * blockDim.x - 1).
This is a standard and reliable pattern for generating a linear index in CUDA, ensuring each thread has a distinct ID that can be used to access global memory safely.

The code below demonstrates a simple kernel that uses this formula to print each thread's global ID. It also contains a minimal host program to launch the kernel. Note that in real applications you would avoid printing from kernels due to performance considerations, but it serves as a clear illustration here.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_global_id()
{
    // Compute the global ID for this thread
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // For demonstration purposes, print the global ID
    // (Printing from kernels is not efficient but fine for small examples)
    printf("Global ID: %d (Block: %d, Thread: %d)\n", global_id, blockIdx.x, threadIdx.x);
}

int main()
{
    // Define block and grid sizes
    const int threadsPerBlock = 8;
    const int numberOfBlocks = 4;

    // Launch the kernel
    print_global_id<<<numberOfBlocks, threadsPerBlock>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
