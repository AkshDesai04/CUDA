/*
Aim of the program:
Modify the program to launch the kernel with two blocks, each with one thread.

Thinking:
The task requires a CUDA C program that demonstrates how to configure a kernel launch
with exactly two blocks and one thread per block. To keep the example minimal yet
illustrative, a simple device kernel that prints its block and thread indices using
`printf` is used. This showcases that the kernel indeed runs with the requested
configuration. The host code performs a single kernel launch with the launch
configuration `<<<2, 1>>>` (two blocks, one thread each). After the launch we call
`cudaDeviceSynchronize()` to ensure that the kernel completes before the program
terminates, which is necessary when using device `printf`. No additional memory
allocation or complex logic is needed for this demonstration. The program is fully
self‑contained and can be compiled with `nvcc` to produce an executable that prints
the expected output when run.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void myKernel(void)
{
    // Use device printf to show which block and thread executed the kernel
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main(void)
{
    // Launch kernel with 2 blocks, each with 1 thread
    myKernel<<<2, 1>>>();

    // Wait for the kernel to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset the device (optional but clean)
    cudaDeviceReset();

    return 0;
}
