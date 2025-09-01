```cpp
/* 
Launch a 3D grid of (2, 2, 2) blocks. Have each thread print its 3D block index (`blockIdx.x`, `blockIdx.y`, `blockIdx.z`). 

To achieve this, we define a simple CUDA kernel that uses `printf` to output the block indices from each thread. We launch the kernel with a grid of dimensions (2, 2, 2). To keep the output manageable and focused on block indices, we use a single thread per block (`blockDim` set to (1, 1, 1)). Each thread will therefore print the block indices of its own block. After launching the kernel we call `cudaDeviceSynchronize()` to ensure all output is flushed before the program exits. Error checking is minimal; for production code you would typically wrap CUDA API calls with error handling macros. 
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void printBlockIdx()
{
    // Each thread prints its block's 3D index
    printf("Thread (%d,%d,%d) in block (%d,%d,%d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z);
}

int main()
{
    // Define grid dimensions (2, 2, 2)
    dim3 gridDim(2, 2, 2);
    // Define block dimensions (1, 1, 1) so each block has a single thread
    dim3 blockDim(1, 1, 1);

    // Launch kernel
    printBlockIdx<<<gridDim, blockDim>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}
```