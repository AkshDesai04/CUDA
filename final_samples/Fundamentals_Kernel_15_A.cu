```c
/*
Aim: Launch the hanging kernel and try to use `nvidia-smi` in another terminal. You should see the GPU utilization at 100%.

Thinking:
1. We need a CUDA program that launches a kernel that never exits, effectively hogging the GPU.
2. The kernel will be an infinite loop `while (true) {}` executed by each thread.
3. To maximize GPU utilization, we launch a large number of threads: 256 blocks with 256 threads each, giving 65,536 concurrent threads.
4. The host program will launch the kernel and then wait on `cudaDeviceSynchronize()`. Since the kernel never finishes, the host will block forever, but the GPU will stay fully busy.
5. The user can open another terminal and run `nvidia-smi` to observe 100% GPU utilization.
6. Basic error checking is included for the kernel launch.
7. The code is written in pure C with CUDA extensions, compiling with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hanging_kernel(void) {
    // Each thread enters an infinite loop to keep the GPU busy
    while (1) {
        // No operation, just loop forever
    }
}

int main(void) {
    // Choose a large grid to saturate the GPU
    dim3 grid(256);      // 256 blocks
    dim3 block(256);     // 256 threads per block

    // Launch the hanging kernel
    hanging_kernel<<<grid, block>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for the kernel (will never finish)
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    // This point will never be reached
    return 0;
}
```