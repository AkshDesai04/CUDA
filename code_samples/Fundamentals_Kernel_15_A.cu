```c
/*
Launch the hanging kernel and try to use `nvidia-smi` in another terminal. You should see the GPU utilization at 100%.

Thinking:
The goal is to create a CUDA program that launches a kernel which never terminates, so that the GPU remains busy indefinitely. 
To achieve this, we define a kernel that enters an infinite loop. We must ensure the compiler does not optimize the loop away. 
An empty loop is usually preserved because it never terminates, but to be extra safe we include a harmless memory fence inside the loop. 
The kernel is launched with a minimal configuration (1 block of 1 thread) to keep the launch simple, and we call `cudaDeviceSynchronize()` in main to block the host until the kernel completes (which never happens). 
When this program runs, the GPU will show 100% utilization in `nvidia-smi` until the program is terminated.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hangingKernel(void) {
    // Infinite loop that keeps the kernel alive
    while (true) {
        // Perform a no-op memory fence to avoid aggressive compiler optimization
        __threadfence();
    }
}

int main(void) {
    // Launch the hanging kernel
    hangingKernel<<<1, 1>>>();

    // Wait indefinitely for the kernel to finish (it never will)
    cudaDeviceSynchronize();

    // This point will never be reached
    printf("Kernel finished (this will never print).\n");
    return 0;
}
```