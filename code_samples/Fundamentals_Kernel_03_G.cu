```cpp
/*
Aim of the program:
Launch with a 1D block of 32 threads. Have each thread print its warp ID (`threadIdx.x / 32`).

Thinking:
- The requirement is to launch a CUDA kernel with a single block of 32 threads.
- Each thread will compute its warp ID using integer division `threadIdx.x / 32`.
- Since a warp on NVIDIA GPUs is 32 threads, with only 32 threads in the block, the warp ID for every thread will be 0. However, the calculation is still performed as requested.
- CUDA allows printing from device code using `printf`, but the output will only appear after a device synchronization on the host.
- The kernel is trivial: it simply prints the thread index and the computed warp ID.
- In the host code, we launch the kernel with one block and 32 threads, then call `cudaDeviceSynchronize()` to ensure that all device output is flushed to the host console before the program exits.
- The code includes minimal headers (`<cstdio>` and CUDA runtime) and checks for errors implicitly by relying on `cudaDeviceSynchronize()` to return any launch errors.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void printWarpId()
{
    int warpId = threadIdx.x / 32;
    printf("Thread %d warp ID %d\n", threadIdx.x, warpId);
}

int main()
{
    // Launch one block with 32 threads
    printWarpId<<<1, 32>>>();
    // Wait for the kernel to finish and flush printf output
    cudaDeviceSynchronize();
    return 0;
}
```