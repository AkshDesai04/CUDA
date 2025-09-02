/*
Aim: Write a parent kernel that launches a simple child kernel (e.g., a `printf`) with a configuration of 1 block and 1 thread.

Thinking:
- Use CUDA dynamic parallelism to allow a kernel to launch another kernel. This requires a GPU with compute capability 3.5 or higher and compiling with `-arch=sm_35` or higher.
- Define a child kernel that simply calls `printf` to output a message.
- Define a parent kernel that launches the child kernel with `<<<1,1>>>`.
- In `main`, launch the parent kernel with the same 1x1 configuration and synchronize.
- Include error checking for kernel launches and synchronization.
- Use standard headers (`cstdio` for `printf` and `iostream` for diagnostics).
- Ensure the file ends with the usual `return 0;` in `main`.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void childKernel()
{
    printf("Hello from child kernel\n");
}

__global__ void parentKernel()
{
    // Launch child kernel with 1 block and 1 thread
    childKernel<<<1, 1>>>();
    // Optional: check for errors from the launch
    // Note: In dynamic parallelism, error checking is done on the child kernel launch via cudaGetLastError()
}

int main()
{
    // Launch parent kernel
    parentKernel<<<1, 1>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error launching parent kernel: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error after synchronization: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Successful completion
    return 0;
}
