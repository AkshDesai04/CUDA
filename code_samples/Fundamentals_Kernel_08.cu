/*
 Launch a kernel with a 2D grid of blocks, for example, 4x4 blocks, each with a single thread. Each thread should print its 2D block index (blockIdx.x, blockIdx.y).

 This program demonstrates how to configure a 2D grid of blocks in CUDA and how each thread can print its block indices. 
 - The kernel is defined as __global__ void printBlockIdx() that calls device printf to output blockIdx.x and blockIdx.y.
 - The kernel is launched with a dim3 grid(4,4) and dim3 block(1). Thus there are 16 threads in total, one in each block.
 - The code uses device printf, which is available on compute capability 2.0 and later; 
   for older GPUs, a different method would be needed.
 - After launching the kernel, cudaDeviceSynchronize() is called to wait for kernel completion.
 - Basic error checking is performed via cudaGetLastError().

 This example is simple but illustrates important concepts: grid/block dimensions, device printf, and synchronization.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that prints the block indices of each thread
__global__ void printBlockIdx()
{
    // Each block has only one thread in this example
    printf("BlockIdx: (%d, %d)\n", blockIdx.x, blockIdx.y);
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",        \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),             \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

int main(void)
{
    // Define a 2D grid of 4x4 blocks, each block with 1 thread
    dim3 gridDim(4, 4);
    dim3 blockDim(1, 1, 1);  // Only one thread per block

    // Launch the kernel
    printBlockIdx<<<gridDim, blockDim>>>();
    CUDA_CHECK(cudaGetLastError());

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reset device (optional)
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
