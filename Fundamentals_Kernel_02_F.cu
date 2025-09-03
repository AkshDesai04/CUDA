```cpp
/*
Aim of the program: Add host-side `printf` statements before and after the kernel launch to observe the non-deterministic ordering of host and device prints.

Thinking process:
- The task is to demonstrate that CUDA kernel launches are asynchronous by default.
- We will print a message from the host before launching the kernel, launch a simple kernel that prints a message from each thread, then print another host message immediately after the launch.
- Because the launch is asynchronous, the host may continue executing and printing before the kernel’s prints have been flushed to the console. Depending on scheduling and buffering, the order can vary between runs.
- After the kernel launch we will call `cudaDeviceSynchronize()` to block the host until the device completes, and then print a final host message to show when the device is done.
- The kernel will be very simple, printing its thread index. The host will print three messages: before launch, after launch, and after synchronization.
- No complex error checking is required for the demonstration, but we will include minimal checks for clarity.
- The code is written in a single .cu file and uses standard CUDA and C++ I/O functions.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void gpuPrintKernel()
{
    // Each thread prints its thread index
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main()
{
    // Host print before kernel launch
    printf("Host: Before kernel launch\n");

    // Asynchronous kernel launch
    gpuPrintKernel<<<1, 4>>>();

    // Host print after kernel launch (still asynchronous)
    printf("Host: After kernel launch (async)\n");

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Host print after synchronization
    printf("Host: After cudaDeviceSynchronize\n");

    return 0;
}
```