/*
Aim of the program:
Explain in a comment what a grid of blocks is in the CUDA execution model.

Thinking:
In CUDA, the execution model is hierarchical. At the top level we have a **grid** of thread blocks. Each grid is a collection of blocks that are launched together when a kernel is invoked. The grid can be one-, two-, or three-dimensional. Within the grid, each block is also a potentially multi-dimensional grouping of threads. The grid of blocks defines the global execution scope: each thread has a unique global index that can be computed from its block index and thread index. The grid's size can be set by the user during kernel launch (e.g., `<<<gridDim, blockDim>>>`). Blocks execute independently and may run concurrently on different streaming multiprocessors (SMs). They also allow the GPU to schedule work in a way that balances load and maximizes resource utilization. The grid abstraction is crucial because it allows the programmer to specify how many independent blocks of work are needed for a problem, and the runtime system decides how to map these blocks to the hardware. In summary, a grid of blocks is the top-level structure that contains all blocks executing a kernel; it determines how the work is partitioned and scheduled across the GPU's compute units.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Dummy kernel that does nothing
__global__ void dummyKernel()
{
    // Each thread can compute its global index if needed
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // For demonstration, we simply print the index (not recommended for large grids)
    if (idx < 10) {
        printf("Thread %d\n", idx);
    }
}

int main()
{
    // Define grid and block dimensions
    dim3 blockDim(256);
    dim3 gridDim(2); // Launch 2 blocks

    // Launch the dummy kernel
    dummyKernel<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();

    return 0;
}
