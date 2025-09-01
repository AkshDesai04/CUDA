```cuda
/* 
Aim of the program: Write a kernel that is launched with a 1D block of 64 threads. Each thread should print its own thread index within the block, which is accessible via `threadIdx.x`.

Thinking:
1. Use a simple CUDA kernel that calls `printf` with `threadIdx.x`. 
2. The kernel will be launched with a single block (`gridDim.x = 1`) and a block size of 64 threads (`blockDim.x = 64`). 
3. After launching the kernel, the host will call `cudaDeviceSynchronize()` to ensure all device prints complete before the program exits. 
4. Include basic error checking for the kernel launch and device synchronization. 
5. Compile with a CUDA-capable compiler (e.g., `nvcc -arch=sm_20`) to support device-side `printf`. 
6. The code is minimal, self-contained, and demonstrates the requested behavior. 
*/

#include <cstdio>
#include <cuda_runtime.h>

// Device kernel that prints the thread index
__global__ void printThreadIdx()
{
    printf("Thread index: %d\n", threadIdx.x);
}

int main()
{
    // Define 1D block with 64 threads
    dim3 blockDim(64);
    dim3 gridDim(1);

    // Launch kernel
    printThreadIdx<<<gridDim, blockDim>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device sync failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
```