/*
Aim of the program:
Launch an 8x8 block. Have only the threads on the first row (`threadIdx.y == 0`) print their `threadIdx.x`.

Thinking:
- The kernel will be launched with a single block of dimensions 8x8.
- Each thread will check if its `threadIdx.y` is 0. Only the first row should print.
- Use `printf` inside the kernel to output the thread's x index.
- After kernel launch, the host will call `cudaDeviceSynchronize()` to ensure all printf outputs are flushed before the program exits.
- Include basic error checking for kernel launch.
- The code will be a self‑contained .cu file that can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_first_row_threads()
{
    if (threadIdx.y == 0)
    {
        printf("Thread x = %d\n", threadIdx.x);
    }
}

int main(void)
{
    dim3 blockDim(8, 8, 1);
    dim3 gridDim(1, 1, 1);

    print_first_row_threads<<<gridDim, blockDim>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for device to finish and flush printf buffer
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronize error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
