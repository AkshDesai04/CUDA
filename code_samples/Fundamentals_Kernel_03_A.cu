/*
Aim: Also print the block's dimension, blockDim.x, alongside the thread index.

Thinking:
- The program is a simple CUDA C example that demonstrates how to print information from within a device kernel.
- The kernel will output each thread's global index and the block dimension (blockDim.x) using device printf.
- To see the output, the host program will launch the kernel with a small grid and block configuration.
- After launching, we call cudaDeviceSynchronize() to ensure that all device output is flushed before the program exits.
- Basic error checking is performed after the kernel launch to catch any launch failures.
- This example uses standard C headers and CUDA runtime API, and can be compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printThreadInfo(void)
{
    // Compute global thread index
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // Print thread index and block dimension
    printf("Thread idx: %d, blockDim.x: %d\n", globalIdx, blockDim.x);
}

int main(void)
{
    // Define grid and block dimensions
    dim3 blockDim(4);     // 4 threads per block
    dim3 gridDim(2);      // 2 blocks

    // Launch kernel
    printThreadInfo<<<gridDim, blockDim>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Synchronize to ensure all prints are flushed
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d, line %d\n", err, __LINE__);
        return 1;
    }

    // Reset device (optional)
    cudaDeviceReset();

    return 0;
}