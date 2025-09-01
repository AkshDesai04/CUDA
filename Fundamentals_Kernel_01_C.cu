/*
Aim of the program:
Use `dim3` variables to define a grid of (4,1,1) and a block of (1,1,1) and launch the kernel.

Thought process:
1. The goal is to demonstrate how to set up and launch a CUDA kernel with specific grid and block dimensions using `dim3`.
2. We will write a simple kernel that prints the block and thread indices to verify that the launch configuration works.
3. Since we only use one thread per block, threadIdx.x will always be 0. The blockIdx.x will range from 0 to 3 because the grid has 4 blocks in the x dimension.
4. The kernel will use `printf` from device code to output information; this requires the CUDA runtime to support device-side printing (available on all recent GPUs).
5. In the host code, we will:
   - Define `dim3 grid(4, 1, 1);`
   - Define `dim3 block(1, 1, 1);`
   - Launch the kernel with `testKernel<<<grid, block>>>();`
   - Call `cudaDeviceSynchronize()` to wait for completion and check for errors.
6. The entire program will be self-contained in a single .cu file, compilable with `nvcc`.
7. No external libraries or helper functions are needed beyond the CUDA runtime and standard C++ headers.

Implementation details:
- Kernel signature: `__global__ void testKernel()`.
- Inside kernel: use `printf("Hello from block %d thread %d\n", blockIdx.x, threadIdx.x);`.
- Error checking on kernel launch and after synchronization.
- Use `cudaGetLastError()` to capture any launch errors.
- Use `std::cout` on host to indicate completion.

*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void testKernel()
{
    // Print the block and thread indices
    printf("Hello from block %d thread %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
    // Define grid and block dimensions
    dim3 grid(4, 1, 1);   // 4 blocks in x dimension
    dim3 block(1, 1, 1);  // 1 thread per block

    // Launch the kernel
    testKernel<<<grid, block>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA sync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Kernel execution completed successfully.\n");
    return 0;
}
