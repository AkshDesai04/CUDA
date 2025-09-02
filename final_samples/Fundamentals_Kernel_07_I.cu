/*
Aim of the program: Modify the kernel so that threads with `threadIdx.x == threadIdx.y` (on the diagonal) print a special message.

Thinking:
- The kernel will run with a 2D block and possibly 2D grid.
- Inside the kernel, we will check if the current thread is on the diagonal of its block: `threadIdx.x == threadIdx.y`.
- If true, we will print a special message indicating that this is a diagonal thread.
- Otherwise, we will print a normal message.
- For demonstration purposes, we will launch a single block of size 4x4 so that there are 16 threads, including diagonal threads (0,0), (1,1), (2,2), (3,3).
- The kernel uses device printf, which is supported on recent GPUs.
- The host code sets up the grid and block dimensions, launches the kernel, synchronizes, and exits.
- Error checking is included to catch CUDA runtime errors.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void diagonal_kernel()
{
    // Calculate global thread indices if needed
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Each thread prints a message
    if (tx == ty) {
        printf("[Block (%d,%d) Thread (%d,%d)] Diagonal thread!\n", bx, by, tx, ty);
    } else {
        printf("[Block (%d,%d) Thread (%d,%d)] Regular thread.\n", bx, by, tx, ty);
    }
}

int main()
{
    // Define block and grid dimensions
    dim3 blockDim(4, 4); // 4x4 threads per block
    dim3 gridDim(1, 1);  // 1 block

    // Launch the kernel
    diagonal_kernel<<<gridDim, blockDim>>>();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA device synchronization error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
