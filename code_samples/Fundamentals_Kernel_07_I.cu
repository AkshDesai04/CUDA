/*
Aim of the program: Modify the kernel so that threads with `threadIdx.x == threadIdx.y` (on the diagonal) print a special message.

Thinking:
1. The requirement is to adjust a CUDA kernel so that any thread whose threadIdx.x equals threadIdx.y (i.e., diagonal threads within a block) prints a unique message.
2. To demonstrate this, the kernel will run with a 2D thread block (e.g., 4x4) and a 2D grid (e.g., 2x2) so that there are multiple blocks and we can observe diagonal behavior across all blocks.
3. Each thread will check `if (threadIdx.x == threadIdx.y)` and, if true, print a message indicating that it is a diagonal thread; otherwise, it will print a normal message. The message includes the thread's coordinates and the block coordinates for clarity.
4. Since printing from device code uses `printf`, the program will compile with `-rdc=true` (required for device printf in older CUDA versions) or with recent CUDA which supports device-side printf without special flags. The code will simply call `printf` inside the kernel.
5. The host code will set up the kernel launch parameters, launch the kernel, synchronize, and handle any errors. A small helper macro `CHECK_CUDA` will be used for error checking.
6. The program is self-contained: it includes the necessary headers, defines the kernel, launches it, synchronizes, and prints a completion message from the host.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",    \
                    #call, __FILE__, __LINE__,                  \
                    cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

// Kernel that prints special message for diagonal threads
__global__ void diagKernel()
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    if (tx == ty)
    {
        printf("[Diagonal] Block (%d,%d) Thread (%d,%d)\n", bx, by, tx, ty);
    }
    else
    {
        printf("Block (%d,%d) Thread (%d,%d)\n", bx, by, tx, ty);
    }
}

int main()
{
    // Define block and grid dimensions
    dim3 blockDim(4, 4); // 4x4 threads per block
    dim3 gridDim(2, 2);  // 2x2 blocks

    // Launch the kernel
    diagKernel<<<gridDim, blockDim>>>();

    // Wait for kernel to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Kernel execution completed.\n");
    return 0;
}
