/*
Launch a 4x4 grid. Have only the block at (0, 0) print a message.

In order to satisfy the requirement of printing a message only from the block at (0, 0) in a 4x4 grid of thread blocks, we use a CUDA kernel that checks the block indices and calls device-side printf only when both blockIdx.x and blockIdx.y are zero.  
The kernel itself does not need any threads to perform work other than the printing, so we launch it with a block dimension of (1,1,1).  
A 4x4 grid is created by setting dim3 grid(4,4).  
After the kernel launch we call cudaDeviceSynchronize() to ensure that all asynchronous device operations complete before the host program exits, allowing the printed message to appear in the standard output.  
For simplicity and clarity, basic error checking after the kernel launch is performed, and the program returns 0 on success.  
The code uses the standard CUDA runtime API and should compile with nvcc on a device supporting device-side printf (compute capability ≥ 2.0).  
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Device kernel that prints a message only from block (0,0)
__global__ void printFromFirstBlock()
{
    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        printf("Hello from block (0,0)!\n");
    }
}

int main()
{
    // Define a 4x4 grid of thread blocks; each block has 1 thread
    dim3 grid(4, 4);
    dim3 block(1, 1);

    // Launch the kernel
    printFromFirstBlock<<<grid, block>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA sync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
