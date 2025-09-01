/*
Aim: Create a second empty kernel with a different name and launch both kernels sequentially from `main`.

Thinking:
- The requirement is to have two empty CUDA kernels with distinct names.
- The kernels should be launched from the host `main` function one after the other.
- Use the simplest grid/block configuration (1 block, 1 thread) because the kernels do nothing.
- Add basic error checking after each kernel launch to ensure correct execution.
- Include the necessary headers (`cuda_runtime.h`) and a simple `cudaDeviceSynchronize` to make sure kernels finish before the program exits.
- The file is a complete .cu source that can be compiled with `nvcc`.
*/

#include <cuda_runtime.h>
#include <stdio.h>

// First empty kernel
__global__ void kernelOne()
{
    // Intentionally left empty
}

// Second empty kernel with a different name
__global__ void kernelTwo()
{
    // Intentionally left empty
}

int main()
{
    // Launch the first kernel
    kernelOne<<<1, 1>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error launching kernelOne: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Synchronize to ensure kernelOne completes before launching the next kernel
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error after kernelOne synchronization: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch the second kernel
    kernelTwo<<<1, 1>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error launching kernelTwo: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Final synchronization before exiting
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error after kernelTwo synchronization: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Both kernels executed successfully.\n");
    return 0;
}
