/*
Aim of the program:
Launch a 4x4 grid. Have only the block at (0, 0) print a message.

Thinking:
To satisfy the requirement, we create a CUDA kernel that prints a message only if the
current block coordinates (blockIdx.x, blockIdx.y) equal (0,0). We launch the kernel
with a grid of dimensions 4x4 and use a minimal block size (1 thread per block) since
we only need one thread to perform the printing. After launching, we synchronize the
device to ensure the message is printed before the program exits. We include minimal
error checking for kernel launch and device synchronization to make the program
robust and easy to compile and run. The code is self‑contained in a single .cu file
and uses the standard CUDA API and stdio for output.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printOnlyZeroBlock()
{
    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        printf("Hello from block (0,0)!\n");
    }
}

int main()
{
    dim3 gridDim(4, 4);
    dim3 blockDim(1, 1);

    printOnlyZeroBlock<<<gridDim, blockDim>>>();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Synchronize to ensure kernel completion before exiting
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset the device (optional, but good practice for standalone programs)
    cudaDeviceReset();

    return 0;
}
