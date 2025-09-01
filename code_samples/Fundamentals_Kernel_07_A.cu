/*
Change the launch configuration to a non-square 2D block, like 16x4 threads, and print the 2D indices.

Thought process:
- The goal is to demonstrate a CUDA kernel launch with a rectangular block shape.
- We'll use a block of 16 threads in the x‑dimension and 4 threads in the y‑dimension.
- The grid can be any size; for demonstration we choose 2 blocks in x and 3 blocks in y.
- Each thread will print its block and thread indices using device `printf`.
- Since `printf` on device requires compute capability 2.x or higher, the code should be compiled with at least `-arch=sm_20`.
- We'll provide a small error‑checking helper macro for CUDA API calls.
- The host code will launch the kernel, synchronize, and then exit.
- The entire code is self‑contained in a single .cu file and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d - %s (%d)\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err), err); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel that prints the 2D indices of each thread and block
__global__ void printIndicesKernel()
{
    // Each thread prints its own block and thread indices
    printf("Block (%d,%d) Thread (%d,%d)\n",
           blockIdx.x, blockIdx.y,
           threadIdx.x, threadIdx.y);
}

int main(void)
{
    // Define block and grid dimensions
    dim3 blockDim(16, 4);   // 16x4 threads per block (rectangular block)
    dim3 gridDim(2, 3);     // 2 blocks in x, 3 blocks in y

    // Launch kernel
    printIndicesKernel<<<gridDim, blockDim>>>();

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up and exit
    return 0;
}
