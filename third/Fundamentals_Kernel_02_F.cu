/*
Add host-side `printf` statements before and after the kernel launch to observe the non-deterministic ordering of host and device prints.
I want to demonstrate that when a CUDA kernel is launched asynchronously, the host’s subsequent code may execute before the device finishes printing its own messages. The program will:
1. Print a message from the host before launching the kernel.
2. Launch a simple kernel that uses device `printf` to output thread information.
3. Immediately print another message from the host after the kernel launch (without waiting for the kernel to finish).
4. Call `cudaDeviceSynchronize()` to ensure the device finishes and the device prints are flushed.
5. Finally, print a host message after synchronization to confirm that all device prints have been completed.
By inspecting the order of the printed lines, one can observe that the host “after launch” message can appear either before or after the device prints, highlighting the non-deterministic execution order between host and device code in CUDA.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that prints a message from each thread
__global__ void printKernel()
{
    printf("Device print from thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main()
{
    // Host message before launching the kernel
    printf("Host before kernel launch\n");

    // Launch the kernel with 1 block and 4 threads
    printKernel<<<1, 4>>>();

    // Host message immediately after launching the kernel (no synchronization yet)
    printf("Host after kernel launch\n");

    // Synchronize to wait for the kernel to finish and flush device printf output
    cudaDeviceSynchronize();

    // Host message after synchronization
    printf("Host after synchronization\n");

    return 0;
}
