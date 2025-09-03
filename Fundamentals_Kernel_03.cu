/*
Aim of the program (word for word from the prompt):
"Write a kernel that is launched with a 1D block of 64 threads. Each thread should print its own thread index within the block, which is accessible via `threadIdx.x`."

Thinking and design notes:
- We will write a simple CUDA program in a single .cu file.
- The kernel will be called `print_thread_index`. Inside the kernel we use `printf` to print the thread index.
- The kernel launch configuration will be a single block (`<<<1, 64>>>`), giving 64 threads in the block.
- After launching the kernel we call `cudaDeviceSynchronize()` to wait for all device threads to finish before the host exits.
- Error checking is added for the kernel launch and synchronization to aid debugging.
- We use `#include <cstdio>` for `printf` on the host side.
- The program can be compiled with `nvcc` and run on any GPU that supports device-side printf (Compute Capability 2.0 or higher).
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void print_thread_index()
{
    // Each thread prints its own threadIdx.x value
    printf("Thread index: %d\n", threadIdx.x);
}

int main()
{
    // Launch one block of 64 threads
    print_thread_index<<<1, 64>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for the kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Clean up and exit
    return 0;
}
