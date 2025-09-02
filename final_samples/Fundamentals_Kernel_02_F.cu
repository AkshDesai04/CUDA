/*
Aim: Add host-side `printf` statements before and after the kernel launch to observe the non-deterministic ordering of host and device prints.

Thinking:
- Use a simple CUDA kernel that prints a message from each thread.
- On the host, print a message before launching the kernel.
- Immediately after the kernel launch, print another message on the host.
- Do not synchronize immediately after launch, to allow the kernel to start executing asynchronously.
- After the host prints, add cudaDeviceSynchronize() to wait for kernel completion.
- Observe that host prints may appear before, after, or interleaved with device prints, demonstrating non-deterministic ordering.
- Keep the code minimal for clarity, using standard includes and error checking if desired.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple CUDA kernel that prints a message from each thread
__global__ void demoKernel()
{
    printf("Device: Hello from thread %d of block %d\n", threadIdx.x, blockIdx.x);
}

int main()
{
    // Host-side print before launching the kernel
    printf("Host: Before kernel launch\n");

    // Launch the kernel with 1 block of 4 threads
    demoKernel<<<1, 4>>>();

    // Host-side print immediately after launching the kernel
    printf("Host: After kernel launch\n");

    // Wait for the kernel to finish to ensure all device prints are flushed
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
