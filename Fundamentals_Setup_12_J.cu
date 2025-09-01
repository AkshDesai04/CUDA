/*
Aim: Compile with the `-v` (verbose) flag to see the internal steps `nvcc` takes, including where it separates host and device code.

Thinking:
- The goal of this program is simply to demonstrate that NVCC splits the source file into host (CPU) and device (GPU) sections.
- We provide a minimal kernel (`dummyKernel`) that does nothing, just to trigger the compiler to generate PTX and compile it to an executable.
- In `main` we launch this kernel and then call `cudaDeviceSynchronize()` to ensure the kernel has finished before printing a message.
- When compiling with `nvcc -v -o demo demo.cu`, the verbose output will show:
    * The preprocessing stage.
    * The separation of host and device code: the host C/C++ code is compiled by the host compiler, while the device code is compiled into PTX or binary by the CUDA compiler.
    * The linking stage where both host and device objects are combined into the final executable.
- This simple program is enough to observe the internal steps of NVCC, and can be expanded if needed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel()
{
    // Empty kernel to trigger device code generation
}

int main()
{
    // Launch the kernel with a single block and single thread
    dummyKernel<<<1, 1>>>();

    // Wait for the kernel to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Kernel launched and completed successfully.\n");
    return 0;
}
