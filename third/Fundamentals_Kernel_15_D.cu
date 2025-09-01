/*
Aim of the program:
Write a kernel that has a very long but finite loop. Call it without `cudaDeviceSynchronize`. 
The host program finishes almost instantly. Now add `cudaDeviceSynchronize`. 
The host program now waits for the kernel to complete.

Thinking process:
1. Define a simple CUDA kernel that executes a tight loop for a very large number of iterations. 
   The loop is finite, but the iteration count is high enough to make the kernel noticeably slow.
2. In the host code, launch the kernel with a minimal launch configuration (1 block, 1 thread).
3. Use a compile-time macro `USE_SYNC` to decide whether to call `cudaDeviceSynchronize()` after launching the kernel.
   - If `USE_SYNC` is defined, the host will wait for the kernel to finish before proceeding.
   - If `USE_SYNC` is not defined, the host will continue immediately, likely exiting before the kernel finishes.
4. Add basic error checking after kernel launch and after synchronization (if performed).
5. Print messages before and after the kernel launch to observe the timing behavior.
6. Compile the code with or without the `-D USE_SYNC` flag to observe the difference in execution time.

Note:
- The long loop counter is of type `long long` and uses 1e9 iterations; adjust if needed.
- The program demonstrates the difference in host behavior with and without synchronization.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void longLoopKernel()
{
    // Very long but finite loop
    for (long long i = 0; i < 1000000000LL; ++i)
    {
        // Do nothing, just burn CPU cycles
    }
}

int main()
{
    printf("Launching kernel...\n");

    // Launch the kernel with 1 block of 1 thread
    longLoopKernel<<<1, 1>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

#ifndef USE_SYNC
    // No synchronization: host continues immediately
    printf("Kernel launched without synchronization. Host may exit now.\n");
#else
    // Synchronize: host waits for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Kernel finished after synchronization.\n");
#endif

    // End of program
    printf("Host program exiting.\n");
    return 0;
}
