/*
Aim: Launch an 8x8 block. Have only the thread at (0, 0) print a message.

Thinking:
1. The program is a simple CUDA example that demonstrates how to launch a kernel with an 8x8 block of threads.
2. Only the thread with coordinates (0,0) should perform a printf to the console. All other threads do nothing.
3. To achieve this, the kernel will check `threadIdx.x` and `threadIdx.y`. If both are zero, the thread will print a message.
4. Device printf requires a compute capability of at least 3.5; we assume the user will compile with an appropriate architecture.
5. The host code will:
   - Define the kernel.
   - Launch it with a single block of size 8x8.
   - Synchronize and check for errors.
   - Clean up (if necessary) and exit.
6. No additional resources or external headers are needed beyond `<stdio.h>` and `<cuda_runtime.h>`.

The resulting .cu file will compile with `nvcc` and when run will output a single line from the (0,0) thread.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void launch8x8Block()
{
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        printf("Hello from thread (0,0) in block (0,0)!\n");
    }
}

int main()
{
    // Launch kernel with a single block of size 8x8
    launch8x8Block<<<1, dim3(8, 8)>>>();

    // Synchronize to ensure kernel has finished before checking errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Kernel launch completed successfully.\n");
    return 0;
}
