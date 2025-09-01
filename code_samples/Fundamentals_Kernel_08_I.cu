```c
/*
Aim of the program: Modify the kernel so that blocks with `blockIdx.x == blockIdx.y` (on the diagonal) print a special message.

Thinking:
1. The original kernel likely iterates over all blocks and threads, printing something like the block and thread indices.
2. We need to change the kernel so that if a block lies on the diagonal of the grid (i.e., its x and y indices are equal), it prints a special message.
3. Device-side printf is used to output messages; this requires compute capability >= 2.0.
4. The kernel will check the condition `blockIdx.x == blockIdx.y` and branch accordingly.
5. For demonstration, we will launch a 4x4 grid of blocks, each block containing a 2x2 set of threads.
6. The kernel will print a normal message for non-diagonal blocks and a special message for diagonal ones.
7. Basic error checking after kernel launch will be included to catch launch failures.
8. The program is self-contained and can be compiled with `nvcc -o diagonal_msg diagonal_msg.cu` and executed on a CUDA-capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints a message for each thread.
// If the block is on the diagonal (blockIdx.x == blockIdx.y), it prints a special message.
__global__ void printKernel()
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (bx == by)
    {
        // Diagonal block
        printf("Diagonal block! [Block (%d,%d), Thread (%d,%d)]\n", bx, by, tx, ty);
    }
    else
    {
        // Normal block
        printf("Block (%d,%d), Thread (%d,%d)\n", bx, by, tx, ty);
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                  \
    do {                                                                 \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

int main(void)
{
    // Define grid and block dimensions
    dim3 blockDim(2, 2);   // 2x2 threads per block
    dim3 gridDim(4, 4);    // 4x4 blocks

    // Launch the kernel
    printKernel<<<gridDim, blockDim>>>();
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Reset the device (optional)
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```