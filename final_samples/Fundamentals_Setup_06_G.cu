/*
Aim of the program:
Explain in a comment the difference between block dimensions and grid dimensions.

Thought process:
The user explicitly wants a comment that explains the difference between block dimensions and grid dimensions in CUDA.  
- Block dimensions refer to the size of a thread block, expressed as the number of threads in each dimension (x, y, z). They determine how many threads are launched per block and are limited by the GPU’s maximum threads per block.  
- Grid dimensions refer to the number of blocks in the grid, expressed as the number of blocks in each dimension (x, y, z). They determine how many blocks are launched to cover the problem domain and are limited by the maximum grid size supported by the device.  
I’ll provide a clear explanation inside a multi‑line comment. The rest of the code will be a minimal, compilable CUDA program that demonstrates launching a kernel with specific block and grid dimensions, but the main focus is on the explanatory comment.  
The code will include:
  - A simple kernel that does nothing significant.
  - A main function that sets up grid and block dimensions.
  - The kernel launch to illustrate the use of those dimensions.

The file will be a .cu file with all the necessary includes and a basic error checking macro.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that does nothing significant
__global__ void dummyKernel()
{
    // Each thread can compute its global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // For demonstration purposes only
    printf("Thread %d in block %d\n", idx, blockIdx.x);
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s at %s:%d - %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main()
{
    // Define block dimensions (threads per block)
    dim3 blockDim(256, 1, 1);  // 256 threads per block along x

    // Define grid dimensions (blocks per grid)
    dim3 gridDim(10, 1, 1);    // 10 blocks along x

    // Launch the kernel
    dummyKernel<<<gridDim, blockDim>>>();

    // Wait for GPU to finish before accessing on host
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Kernel launch completed.\n");
    return 0;
}
