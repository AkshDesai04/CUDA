/*
Launch a kernel with a 2D block of threads, for example, 8x8 threads. Each thread should print its 2D thread index (`threadIdx.x`, `threadIdx.y`).

The program demonstrates a simple CUDA kernel that uses a 2‑D thread block of size 8×8. 
We define a kernel `printThreadIndices` that contains a `printf` statement to output the x and y coordinates of each thread. 
The host code sets the block dimensions to (8, 8) and launches a single block (grid dimensions of (1, 1)). 
After the kernel launch we call `cudaDeviceSynchronize()` to wait for the kernel to finish and to flush all device prints to the console.

Device `printf` requires the architecture to support it; therefore this code should be compiled with an architecture of at least `sm_20` (e.g., using `nvcc -arch=sm_20`).

No additional memory allocation or error checking is necessary for this simple demonstration, but error checks are included to make the code robust and to illustrate typical CUDA programming practices.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printThreadIndices()
{
    // Each thread prints its 2D thread index
    printf("Thread (%d, %d) in block (%d, %d)\n",
           threadIdx.x, threadIdx.y,
           blockIdx.x, blockIdx.y);
}

int main()
{
    // Define block dimensions (8x8 threads)
    dim3 blockDim(8, 8);
    // Define grid dimensions (1x1 block)
    dim3 gridDim(1, 1);

    // Launch the kernel
    printThreadIndices<<<gridDim, blockDim>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
