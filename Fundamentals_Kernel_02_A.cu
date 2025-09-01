```cuda
/*
Aim: Launch the printing kernel with one block of 10 threads. Observe the output.

Thinking:
1. We need a simple CUDA program that launches a kernel with a single block containing
   10 threads. The kernel will use device-side printf to output the thread index so that
   the user can observe the ordering and number of prints.
2. The kernel signature will be `__global__ void printKernel()`.
3. Inside the kernel, each thread will call `printf("Thread %d running on block %d, thread %d\n",
      blockIdx.x, threadIdx.x, threadIdx.x);`. The `blockIdx.x` will be 0 for the single block,
   and `threadIdx.x` will range from 0 to 9.
4. In `main`, we will:
   - Call the kernel with launch configuration `<<<1, 10>>>`.
   - Call `cudaDeviceSynchronize()` to wait for the kernel to finish and to ensure that
     all device printf output is flushed to the host console.
   - Check for any launch errors with `cudaGetLastError()`.
5. Add minimal includes: `<cstdio>` for `printf` on the host and `cuda_runtime.h` for CUDA
   runtime API. We also include `<iostream>` for optional error printing.
6. Compile the file with `nvcc -o print_kernel print_kernel.cu` and run `./print_kernel` to
   see the output.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void printKernel()
{
    // Each thread prints its own index and the block index
    printf("Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
    // Launch the kernel with 1 block of 10 threads
    printKernel<<<1, 10>>>();

    // Synchronize to ensure all printf calls complete before exiting
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Check for any launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
```