```cuda
/*
Aim of the program:
Launch a normal, fast kernel, followed immediately by the hanging kernel. What is the program's behavior?

Thinking process:
- We need two kernels: one that runs quickly and one that never terminates (hangs).
- CUDA launches kernels asynchronously with respect to the host, but kernels on a single GPU stream execute serially.
- When we launch the fast kernel first, it will execute and finish before the hanging kernel begins.
- The hanging kernel contains an infinite loop, so it never returns control to the host.
- If the host calls cudaDeviceSynchronize() after launching the hanging kernel, it will block indefinitely.
- If we don't synchronize, the program may exit before the kernel executes, but the device will still be busy.
- For demonstration, we will print a message after each launch and after the sync call to show where the program stops.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Fast kernel: each thread writes to an output array
__global__ void fastKernel(int *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = idx;
    }
}

// Hanging kernel: infinite loop
__global__ void hangingKernel() {
    // All threads spin forever
    while (true) {
        // Prevent compiler optimizations
        asm volatile("" ::: "memory");
    }
}

int main(void) {
    const int N = 1024;
    int *d_output = nullptr;
    size_t size = N * sizeof(int);

    // Allocate device memory
    cudaMalloc((void**)&d_output, size);

    // Launch the fast kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    fastKernel<<<gridSize, blockSize>>>(d_output, N);
    printf("Fast kernel launched.\n");

    // Launch the hanging kernel immediately after
    hangingKernel<<<1, 1>>>();
    printf("Hanging kernel launched.\n");

    // Synchronize to see that the program will block here
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("After sync.\n");
    }

    // Clean up (unreachable due to hanging kernel)
    cudaFree(d_output);
    return 0;
}
```