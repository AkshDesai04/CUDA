/*
Aim of the program:
Combine the concepts of the previous two questions. Launch a kernel with a grid of 4 blocks, each containing 16 threads. Each thread must calculate and print its unique 'global ID' using the formula: `int id = blockIdx.x * blockDim.x + threadIdx.x;`.

Thinking:
1. We need a simple CUDA program that launches a kernel with 4 blocks and 16 threads per block.
2. Each thread will compute its global ID with the provided formula.
3. We will use device-side printf to output the ID. This requires a compute capability of at least 2.0, but for typical GPUs it will work.
4. The kernel is trivial, so we can inline it as a __global__ function.
5. In main, we set the grid and block dimensions, launch the kernel, and synchronize the device to ensure all prints are flushed before the program exits.
6. Include necessary headers: <stdio.h> for printf and <cuda_runtime.h> for CUDA runtime API.
7. No additional libraries or files are required. The code is self-contained in a single .cu file.
8. We keep the code minimal and add comments for clarity, but adhere strictly to the requirement of outputting only the code file content.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the unique global ID of each thread
__global__ void print_global_id()
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d: Global ID = %d\n", threadIdx.x, id);
}

int main(void)
{
    // Define grid and block dimensions
    dim3 blockDim(16); // 16 threads per block
    dim3 gridDim(4);   // 4 blocks

    // Launch the kernel
    print_global_id<<<gridDim, blockDim>>>();

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Reset the device (optional)
    cudaDeviceReset();

    return 0;
}
