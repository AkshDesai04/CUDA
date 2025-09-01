/*
Launch a 3D block of (4, 4, 4) threads. Have each thread print its 3D thread index (`threadIdx.x`, `threadIdx.y`, `threadIdx.z`).

Thinking:
The goal is to write a CUDA program that launches a single block with a 3‑D grid of threads
(4 threads along each dimension). Each thread will output its 3‑D index using the CUDA
device printf function. 
We will create a kernel `print_thread_idx` that simply calls `printf` with the thread indices.
The host will launch this kernel with `<<<1, dim3(4,4,4)>>>
` and then call `cudaDeviceSynchronize` to ensure all device output is flushed before the
program exits. Error checking will be minimal for clarity, but we’ll still verify the kernel
launch and device sync return status. This program is intentionally simple and self‑contained,
ready to compile with `nvcc -arch=sm_20 example.cu` (or any newer architecture). 
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_thread_idx(void)
{
    // Each thread prints its 3D index
    printf("Thread (%d, %d, %d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(void)
{
    // Define block dimensions: 4x4x4 threads
    dim3 blockDim(4, 4, 4);
    dim3 gridDim(1, 1, 1); // Single block

    // Launch the kernel
    print_thread_idx<<<gridDim, blockDim>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Synchronize to ensure all prints are done before program exits
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device sync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
