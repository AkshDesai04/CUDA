/*
Launch with 16 blocks. Have each thread print its block ID multiplied by 10.

Thinking:
- The task requires launching a kernel with 16 blocks.
- Each thread should print its block ID times 10. If we have more than one thread per block,
  all those threads would print the same block ID times 10. To avoid duplicate prints, we
  choose one thread per block (threadIdx.x == 0).
- Use device printf (requires compute capability >= 2.0).
- Kernel: __global__ void printBlockIdTimes10()
- In kernel, if (threadIdx.x == 0) printf("%d\n", blockIdx.x * 10);
- In host, launch with <<<16,1>>>.
- Call cudaDeviceSynchronize() to wait for prints to finish.
- Include basic error checking after kernel launch.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printBlockIdTimes10(void)
{
    // Only the first thread in each block prints to avoid duplicate outputs.
    if (threadIdx.x == 0)
    {
        printf("%d\n", blockIdx.x * 10);
    }
}

int main(void)
{
    // Launch the kernel with 16 blocks and 1 thread per block.
    printBlockIdTimes10<<<16, 1>>>();

    // Check for launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the device to finish before accessing the results.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
