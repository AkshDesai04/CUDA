```cpp
/*
Aim of the program:
Launch the kernel with a 2D block of (8, 8) threads and have each thread print only its `threadIdx.x`.

Thinking:
- The kernel needs to be launched with a block of dimension 8x8, so the blockDim is set accordingly.
- Since we only care about printing `threadIdx.x`, each thread will simply call `printf` with that value.
- CUDA requires enabling device printf; we just use the standard `printf` inside the kernel.
- The host code will set up the launch configuration, invoke the kernel, synchronize, and check for errors.
- We keep the code minimal: one kernel, one launch, no grid dimension since one block suffices.
- Include necessary headers: <cstdio> for printf, <cuda_runtime.h> for CUDA API.
- Use `cudaDeviceSynchronize()` to ensure all device prints are flushed before the program exits.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void printThreadIdxX()
{
    // Each thread prints its threadIdx.x
    printf("threadIdx.x = %d\n", threadIdx.x);
}

int main()
{
    // Define block dimension (8, 8)
    dim3 blockDim(8, 8);
    // Launch with one block
    printThreadIdxX<<<1, blockDim>>>();

    // Wait for the kernel to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
```