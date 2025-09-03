```cpp
/*
Use `dim3` variables to define a grid of (4,1,1) and a block of (1,1,1) and launch the kernel.

Thought process:
1. The user wants a simple CUDA program that demonstrates how to set up a grid and block using dim3.
2. We need to create a .cu file that compiles and runs on a CUDA-capable device.
3. The grid dimension should be (4,1,1) and the block dimension should be (1,1,1), so there will be 4 total threads.
4. The kernel can simply print the thread index using device-side printf for clarity.
5. After launching the kernel, the host will call cudaDeviceSynchronize() to ensure all output is flushed before the program exits.
6. Include the necessary headers (`<cstdio>`, `<cuda_runtime.h>`).
7. Provide basic error checking for the kernel launch.
8. The code will be self‑contained and ready to compile with `nvcc`.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void simpleKernel()
{
    // Each thread prints its unique ID
    printf("Thread %d running on block (%d,%d,%d) in grid (%d,%d,%d)\n",
           threadIdx.x,
           blockIdx.x, blockIdx.y, blockIdx.z,
           gridDim.x, gridDim.y, gridDim.z);
}

int main()
{
    // Define grid and block dimensions using dim3
    dim3 gridDim(4, 1, 1);   // (4,1,1)
    dim3 blockDim(1, 1, 1);  // (1,1,1)

    // Launch the kernel
    simpleKernel<<<gridDim, blockDim>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}
```