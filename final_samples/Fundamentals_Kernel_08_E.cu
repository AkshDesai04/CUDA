```cuda
/*
Aim of the program:
Combine with 2D blocks. Launch a 2x2 grid of 4x4 blocks. Have each thread calculate and print its global 2D index: `int global_x = blockIdx.x * blockDim.x + threadIdx.x;` and `int global_y = ...`.

Thinking process:
1. The requirement is to create a simple CUDA kernel that runs on a 2D grid with 2 blocks along each dimension (x and y), and each block has a 4x4 2D arrangement of threads.
2. Inside the kernel, each thread should compute its global 2D coordinates using the standard CUDA formula:
   - global_x = blockIdx.x * blockDim.x + threadIdx.x
   - global_y = blockIdx.y * blockDim.y + threadIdx.y
3. The kernel will use `printf` to output these coordinates. `printf` in device code requires a compute capability of at least 2.0, which is supported by most modern GPUs.
4. The host code will set up the kernel launch parameters, invoke the kernel, and synchronize the device to ensure all output is flushed before the program exits.
5. Basic error checking after the kernel launch will be included for robustness.
6. The program will be a single .cu file containing all the necessary includes, the kernel definition, and the `main` function.

*/

// Necessary includes
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel to print global 2D indices of each thread
__global__ void print_global_indices()
{
    // Compute global indices
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Print global indices
    printf("Thread (%d, %d) => Global Index (%d, %d)\n",
           threadIdx.x, threadIdx.y, global_x, global_y);
}

int main(void)
{
    // Define grid and block dimensions
    dim3 gridDim(2, 2);    // 2 blocks in x and y
    dim3 blockDim(4, 4);   // 4 threads in x and y per block

    // Launch the kernel
    print_global_indices<<<gridDim, blockDim>>>();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for the device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Optional: reset device
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
```