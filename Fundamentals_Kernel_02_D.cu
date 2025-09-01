/*
Aim of the program:
Modify the `printf` to include the block's ID: `printf("Hello from block %d!\n", blockIdx.x);`. Launch with 8 blocks.

Thinking:
1. The requirement is to create a CUDA kernel that prints a message containing the block index (`blockIdx.x`).
2. The kernel will be launched with 8 blocks. The number of threads per block can be minimal (e.g., 1 thread per block) because we only need one print per block.
3. Use `printf` inside the kernel; CUDA supports `printf` for device code.
4. After kernel launch, call `cudaDeviceSynchronize()` to wait for all prints to finish before exiting the host program.
5. Provide basic error checking for the kernel launch and device synchronization to help debug if needed.
6. The code is written as a complete `.cu` file with a `main` function and a kernel definition, ready to compile with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the block ID
__global__ void helloKernel()
{
    // Print the block's ID
    printf("Hello from block %d!\n", blockIdx.x);
}

int main()
{
    // Number of blocks and threads per block
    const int numBlocks = 8;
    const int threadsPerBlock = 1;

    // Launch the kernel
    helloKernel<<<numBlocks, threadsPerBlock>>>();

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching helloKernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error during cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
