/*
Call `cudaDeviceSynchronize()` after the kernel launch. Does this guarantee the `printf` output will appear before subsequent host code executes? Explain in a comment. (Answer: No, `printf` is buffered).

Thinking:
- The CUDA runtime collects device-side printf output in a buffer that is transferred back to the host only when the kernel completes.
- `cudaDeviceSynchronize()` guarantees that the kernel has finished executing, but it does not flush the device printf buffer to the host's stdout immediately.
- The host program may continue executing and may perform its own prints or other work before the device printf output is actually written to the console.
- Therefore, even after calling `cudaDeviceSynchronize()`, the device printf output is not guaranteed to appear before subsequent host code execution. It will be printed when the buffer is flushed (typically at program termination or when the host flushes stdout), but this flush is independent of `cudaDeviceSynchronize()`.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple kernel that prints from device
__global__ void kernel()
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Device: thread %d\n", idx);
}

int main()
{
    // Launch kernel with 128 threads
    kernel<<<4, 32>>>();
    
    // Synchronize device to ensure kernel has finished
    cudaDeviceSynchronize();

    // Host code executes immediately after synchronization
    printf("Host: code executed after device synchronization.\n");

    // Clean up
    cudaDeviceReset();
    return 0;
}
