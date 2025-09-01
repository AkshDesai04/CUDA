```c
/*
Aim of the program:
Declare `dim3 grid(4);` and `dim3 block(64);`. Launch a kernel and print
`gridDim.x` and `blockDim.x` from within the kernel to verify they match.

Thinking:
1. We need a simple CUDA program that declares a grid with 4 blocks in
   the x-dimension and a block with 64 threads in the x-dimension.
2. The kernel will access the built‑in variables `gridDim.x` and
   `blockDim.x` to get the dimensions of the current execution context.
3. To avoid flooding the output, we’ll only have a single thread (e.g.,
   threadIdx.x==0 && blockIdx.x==0) perform the print. If we printed from
   every thread we would get 4 * 64 lines, which is unnecessary.
4. We’ll use `printf` inside the kernel. Since CUDA 3.0+ the device
   `printf` works when compiled with `-arch=sm_20` or later. We'll
   compile with `nvcc` which defaults to a recent architecture.
5. After launching the kernel, the host code will call
   `cudaDeviceSynchronize()` to ensure the kernel finishes before the
   program exits.
6. Basic error checking will be added for the kernel launch and
   synchronization to make the program robust.

The code below follows these steps and compiles to a single `.cu` file.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void printDimsKernel() {
    // Only the first thread of the first block prints the dimensions
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("gridDim.x = %d, blockDim.x = %d\n", gridDim.x, blockDim.x);
    }
}

int main(void) {
    // Declare grid and block dimensions
    dim3 grid(4);   // 4 blocks in x-dimension
    dim3 block(64); // 64 threads in x-dimension per block

    // Launch kernel
    printDimsKernel<<<grid, block>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Device synchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset device (optional but good practice)
    cudaDeviceReset();

    return 0;
}
```