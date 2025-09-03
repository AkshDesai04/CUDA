```cpp
/*
Aim of the program: Declare `dim3 grid(4);` and `dim3 block(64);`. Launch a kernel and print `gridDim.x` and `blockDim.x` from within the kernel to verify they match.

Thinking process:
- We need a minimal CUDA C program that sets up a grid of 4 blocks and a block of 64 threads.
- The kernel should print the values of gridDim.x and blockDim.x as seen by the device.
- Since each thread would print the same values, we restrict printing to a single thread (e.g., threadIdx.x == 0) to keep the output readable.
- Use device-side printf, which requires the target GPU architecture to support it (compute capability >= 2.0). The standard CUDA runtime header and C++ i/o header are included.
- The main function will launch the kernel with the specified launch configuration, then synchronize and check for errors.
- Finally, we output a simple success message from the host.
*/

#include <cuda_runtime.h>
#include <cstdio>

__global__ void print_grid_block_info()
{
    if (threadIdx.x == 0) {
        printf("Inside kernel: gridDim.x = %d, blockDim.x = %d\n",
               gridDim.x, blockDim.x);
    }
}

int main()
{
    dim3 grid(4);   // 4 blocks in the grid
    dim3 block(64); // 64 threads per block

    // Launch the kernel
    print_grid_block_info<<<grid, block>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Kernel execution completed successfully.\n");
    return 0;
}
```