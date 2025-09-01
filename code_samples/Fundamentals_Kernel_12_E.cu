```cpp
/*
Aim: Declare dim3 grid(2, 2); and dim3 block(8, 8);. Launch a kernel and print the x and y components of gridDim and blockDim.

Thinking:
- We need a CUDA C program that sets up a grid of size 2x2 and a block of size 8x8.
- The kernel should print the x and y components of both gridDim and blockDim.
- Using printf inside a CUDA kernel is straightforward, but printing from every thread would produce a lot of output.
- To keep the output concise, each block will print once from the thread with (threadIdx.x==0 && threadIdx.y==0).
- The kernel will be simple, with no memory allocation or complex logic.
- After kernel launch, we synchronize to ensure the output is flushed before the program exits.
- Include error checking for kernel launch and device synchronization for good practice.
- The final code is a self-contained .cu file that can be compiled with nvcc and executed on a CUDA-capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the x and y components of gridDim and blockDim.
// Each block prints once from its first thread to avoid clutter.
__global__ void printDim()
{
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("gridDim: (%d, %d), blockDim: (%d, %d)\n",
               gridDim.x, gridDim.y,
               blockDim.x, blockDim.y);
    }
}

int main(void)
{
    // Define grid and block dimensions as specified.
    dim3 grid(2, 2);
    dim3 block(8, 8);

    // Launch the kernel.
    printDim<<<grid, block>>>();

    // Check for kernel launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Synchronize to ensure all printf outputs are flushed.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Success
    return EXIT_SUCCESS;
}
```