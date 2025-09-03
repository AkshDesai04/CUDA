```cpp
/*
Aim of the program:
Declare `dim3 grid(2, 2);` and `dim3 block(8, 8);`. Launch a kernel and print the `x` and `y` components of `gridDim` and `blockDim`.

Thinking:
1. Include necessary headers: <stdio.h> for printf and <cuda_runtime.h> for CUDA types and functions.
2. Define a global kernel `printDims` that uses `printf` to output `gridDim.x`, `gridDim.y`, `blockDim.x`, and `blockDim.y`.
3. In `main`, set up the grid and block dimensions as specified: `dim3 grid(2, 2);` and `dim3 block(8, 8);`.
4. Launch the kernel with these dimensions: `printDims<<<grid, block>>>( );`.
5. Synchronize the device with `cudaDeviceSynchronize()` to ensure all prints are flushed before the program exits.
6. Return 0.

The output will be printed from each thread, but for clarity, we only need one print; however, as requested, the kernel prints the dimensions, so each thread will print the same values. For brevity, we can restrict to threadIdx.x == 0 && threadIdx.y == 0 to avoid duplicated prints. But the problem statement didn't restrict, so we will keep it simple and print once per thread. This will produce many identical lines, which is acceptable for demonstration.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printDims()
{
    // Print grid and block dimensions
    printf("gridDim.x = %d, gridDim.y = %d\n", gridDim.x, gridDim.y);
    printf("blockDim.x = %d, blockDim.y = %d\n", blockDim.x, blockDim.y);
}

int main()
{
    dim3 grid(2, 2);
    dim3 block(8, 8);

    printDims<<<grid, block>>>();
    cudaDeviceSynchronize();

    return 0;
}
```