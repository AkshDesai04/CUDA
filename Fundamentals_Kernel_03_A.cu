/*
Also print the block's dimension, `blockDim.x`, alongside the thread index.
I approached this task by first interpreting the requirement: we need a simple CUDA program that demonstrates how to access the built‑in CUDA variables `threadIdx.x` (the thread's local index) and `blockDim.x` (the size of the block in the X dimension). The aim is to print these values from each thread. 

To keep the program self‑contained and easy to compile, I chose to use `printf` from the device side. `printf` is supported on most recent CUDA devices and allows straightforward output without needing to copy data back to the host. The program sets up a small grid of blocks and threads (2 blocks, 4 threads each) to keep the output readable. After launching the kernel, the host calls `cudaDeviceSynchronize()` to ensure all output has been flushed before the program exits. Basic error checking macros are included to catch any launch or runtime errors.

The code is written in CUDA C, ready to be compiled with `nvcc`. No external dependencies are required beyond the CUDA toolkit. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for error checking */
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

/* Kernel that prints thread index and block dimension in X */
__global__ void print_kernel()
{
    /* Thread index within the block */
    int thread_idx = threadIdx.x;
    /* Size of the block in X dimension */
    int block_dim_x = blockDim.x;

    /* Device printf (supported on compute capability >= 2.0) */
    printf("Thread %d of block %d: blockDim.x = %d\n",
           thread_idx, blockIdx.x, block_dim_x);
}

int main(void)
{
    /* Define block and grid dimensions */
    dim3 threads_per_block(4);
    dim3 num_blocks(2);

    /* Launch kernel */
    print_kernel<<<num_blocks, threads_per_block>>>();
    CUDA_CHECK(cudaGetLastError());

    /* Wait for GPU to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}
