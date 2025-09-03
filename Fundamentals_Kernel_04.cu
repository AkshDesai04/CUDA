```cuda
/*
Aim of the program:
Write a kernel that is launched with a 1D grid of 8 blocks, where each block has only one thread. Each thread should print its own block index, which is accessible via `blockIdx.x`.

Thinking:
1. We need a device kernel that uses the built-in variable blockIdx.x to identify the block of the current thread.
2. Since each block contains only one thread, the threadIdx.x will always be 0, so we can ignore it.
3. The kernel will call `printf` to output the block index. Device printf requires the host to call `cudaDeviceSynchronize()` after the kernel launch to ensure all output is flushed.
4. The host code will set up the kernel launch configuration to have 8 blocks with 1 thread per block: `kernel<<<8, 1>>>();`.
5. Include necessary headers (`stdio.h` for printf, `cuda_runtime.h` for CUDA API functions).
6. Compile the code with a CUDA compiler such as `nvcc -arch=sm_35 test.cu -o test` (architecture can be adjusted as needed).
7. The program is self-contained and will compile to a .cu file.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the block index of each thread
__global__ void printBlockIndex()
{
    // Each block has a single thread, so threadIdx.x is always 0
    // blockIdx.x gives the block index within the grid
    printf("Block index: %d\n", blockIdx.x);
}

int main()
{
    // Launch the kernel with 8 blocks, 1 thread per block
    printBlockIndex<<<8, 1>>>();

    // Wait for the kernel to finish and flush the device printf output
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Optionally reset the device (not strictly necessary)
    cudaDeviceReset();

    return 0;
}
```