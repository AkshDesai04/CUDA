/*
Aim of the program:
Launch an 8x8 block. Have only the thread at (0, 0) print a message.

Thoughts and approach:
- We need a CUDA kernel that runs on an 8x8 block of threads.
- Inside the kernel, we check the thread coordinates (threadIdx.x, threadIdx.y).
- If both are zero, we use printf to output a message. All other threads do nothing.
- The host code will launch the kernel with a single block of size 8x8.
- After launching, we synchronize the device to ensure all output is flushed.
- No additional memory or complex logic is needed. The program is minimal but demonstrates thread indexing and device printf.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void launchBlock()
{
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        printf("Hello from thread (0,0) in 8x8 block!\n");
    }
}

int main()
{
    // Define block dimensions: 8x8
    dim3 blockDim(8, 8);
    dim3 gridDim(1, 1); // single block

    // Launch kernel
    launchBlock<<<gridDim, blockDim>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset device (optional)
    cudaDeviceReset();

    return 0;
}
