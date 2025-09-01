/*
Aim of the program: Modify the program to launch the kernel with two blocks, each with one thread.

Thinking:
1. The goal is to demonstrate a CUDA kernel launch that uses 2 blocks, with each block containing a single thread.
2. Use dim3 types for grid and block dimensions:
   - blockDim = dim3(1, 1, 1)  // one thread per block
   - gridDim  = dim3(2, 1, 1)  // two blocks
3. In the kernel, we compute a global thread index to identify each thread, though here each block has only one thread, so threadIdx.x will always be 0.
4. Use printf inside the kernel to output the block and thread indices, allowing us to verify that the kernel runs on the intended configuration.
5. In the host code, launch the kernel with the specified grid and block dimensions and synchronize the device to ensure the kernel completes before the program exits.
6. Basic error checking could be added after the kernel launch, but for simplicity, only cudaDeviceSynchronize() is used.

The resulting program is a minimal, self‑contained CUDA C++ file that can be compiled with nvcc and executed to observe two kernel launches, one per block.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that prints its block and thread indices
__global__ void printIndicesKernel()
{
    // Compute global thread index (though threadIdx.x will always be 0)
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from global thread %d (block %d, thread %d)\n",
           globalIdx, blockIdx.x, threadIdx.x);
}

int main()
{
    // Launch configuration: 2 blocks, each with 1 thread
    dim3 gridDim(2, 1, 1);   // two blocks
    dim3 blockDim(1, 1, 1);  // one thread per block

    // Launch the kernel
    printIndicesKernel<<<gridDim, blockDim>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
