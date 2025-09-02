```cuda
/*
Print the values of `blockDim.x` and `gridDim.x` from within the kernel.

The goal is to demonstrate how to access and display the block and grid
dimensions from inside a CUDA kernel.  In CUDA, each thread can query
the built‑in variables blockDim, gridDim, blockIdx, threadIdx, etc.
Here we simply launch a single block with a single thread and let that
thread print the desired values using the device printf facility.
This keeps the output concise and illustrates the concepts without
producing a large amount of output.

Key points considered:
- Use `__global__` to define a kernel.
- Use `printf` from the device; requires compute capability >= 2.0
  and the target system must support device-side printf.
- Synchronize the host after kernel launch to ensure the prints are
  flushed before the program exits.
- Add basic error checking for the kernel launch and device sync.
- Keep the program minimal, focusing only on printing blockDim.x
  and gridDim.x.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that prints blockDim.x and gridDim.x
__global__ void printDims()
{
    // Only one thread prints to avoid duplicated output
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("blockDim.x = %d, gridDim.x = %d\n",
               blockDim.x, gridDim.x);
    }
}

int main()
{
    // Launch one block with one thread
    printDims<<<1, 1>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for kernel to finish and flush device printf output
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```