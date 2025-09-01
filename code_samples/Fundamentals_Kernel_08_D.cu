/*
Launch a 3D grid of (2, 2, 2) blocks. Have each thread print its 3D block index (blockIdx.x, blockIdx.y, blockIdx.z).

Thinking process:
1. The program should launch a 3D grid with dimensions (2, 2, 2). 
2. Each block will have a modest number of threads (2×2×2) to keep the output readable while still demonstrating multiple threads per block. 
3. Each thread should output the block index it belongs to. Including thread indices in the print statement helps confirm that each thread is indeed executing and that the block indices are constant within a block.
4. Device printf is used to print from GPU kernels. 
5. Error checking is performed after the kernel launch and after cudaDeviceSynchronize.
6. The code is written in a single .cu file, with includes, kernel definition, main function, and proper cleanup.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printBlockIdx()
{
    // Each thread prints its block index
    printf("Thread (%d, %d, %d) in Block (%d, %d, %d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z);
}

int main()
{
    // Define a 3D grid of (2, 2, 2) blocks
    dim3 gridDim(2, 2, 2);
    // Define a small number of threads per block (2×2×2)
    dim3 blockDim(2, 2, 2);

    // Launch the kernel
    printBlockIdx<<<gridDim, blockDim>>>();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d: %s\n", err, cudaGetErrorString(err));
        return 1;
    }

    // Reset the device (optional)
    cudaDeviceReset();

    return 0;
}
