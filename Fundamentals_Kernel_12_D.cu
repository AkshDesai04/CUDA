/*
Declare `dim3 grid(4);` and `dim3 block(64);`. Launch a kernel and print `gridDim.x` and `blockDim.x` from within the kernel to verify they match.

Thinking:
- The program is a simple CUDA C example that demonstrates how to set up grid and block dimensions.
- We declare a grid with 4 blocks in the x dimension and a block with 64 threads in the x dimension.
- The kernel will be launched with these dimensions and will print out the values of `gridDim.x` and `blockDim.x` using the device printf function.
- To avoid excessive output, only one thread per block (threadIdx.x == 0) will perform the print.
- After launching the kernel, the host code will synchronize the device and check for errors.
- The code uses standard CUDA error checking macro for simplicity.
- The program should compile with nvcc and run on a GPU that supports device printf (compute capability 2.0 or higher).
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printDimsKernel()
{
    if (threadIdx.x == 0) {
        // Print grid and block dimensions
        printf("Block %d/%d: gridDim.x = %d, blockDim.x = %d\n",
               blockIdx.x, gridDim.x, gridDim.x, blockDim.x);
    }
}

int main()
{
    // Define grid and block dimensions
    dim3 grid(4);   // 4 blocks in x-dimension
    dim3 block(64); // 64 threads in x-dimension

    // Launch the kernel
    printDimsKernel<<<grid, block>>>();
    // Synchronize to ensure kernel has finished
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset device (optional)
    cudaDeviceReset();

    return 0;
}
