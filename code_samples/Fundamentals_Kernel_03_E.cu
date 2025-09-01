/*
Aim of the program: 
Modify the kernel so that only threads in the first half of the block (i.e., `threadIdx.x < blockDim.x / 2`) print their ID.

Thinking:
1. We need a simple CUDA kernel that prints the thread ID using `printf`.
2. To satisfy the requirement, we add a conditional check inside the kernel: if `threadIdx.x` is less than half the block dimension, then call `printf`.
3. In the host code, we launch the kernel with a single block containing a reasonable number of threads (e.g., 256).
4. After launching, we synchronize the device and reset it to clean up.
5. The code should compile with `nvcc` and run on a CUDA-capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printFirstHalfThreadIds()
{
    // Only threads in the first half of the block print their ID
    if (threadIdx.x < blockDim.x / 2)
    {
        printf("Thread ID: %d\n", threadIdx.x);
    }
}

int main()
{
    // Define block and grid dimensions
    const int threadsPerBlock = 256;
    const int blocksPerGrid = 1;

    // Launch the kernel
    printFirstHalfThreadIds<<<blocksPerGrid, threadsPerBlock>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset the device (optional but good practice)
    cudaDeviceReset();

    return 0;
}
