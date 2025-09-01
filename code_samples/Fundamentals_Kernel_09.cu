/*
Create a `__device__` function called `square_me` that takes an integer `x` and returns `x*x`. Create a `__global__` kernel that calls this `square_me` function on its `threadIdx.x` and prints the result.

Thinking:
- The device function `square_me` is simple: it returns the square of its integer argument. It will be called from a kernel, so it must be marked `__device__`.
- The kernel will launch with a sufficient number of threads so each thread can compute its own `threadIdx.x`. Inside the kernel, we compute the square by calling `square_me` and then use `printf` (available on device from compute capability 2.0 onward) to print the thread id, the original value, and the squared result.
- The host `main` function will set up the launch configuration (e.g., one block with 16 threads), launch the kernel, and then call `cudaDeviceSynchronize()` to wait for completion.
- We include basic error checking for the kernel launch and synchronization.
- Standard headers `<stdio.h>` and `<cuda_runtime.h>` are included. The code compiles as a .cu file and can be built with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__device__ int square_me(int x) {
    return x * x;
}

__global__ void square_kernel() {
    int x = threadIdx.x;
    int result = square_me(x);
    printf("Thread %d: %d squared is %d\n", x, x, result);
}

int main() {
    // Launch kernel with 1 block of 16 threads
    square_kernel<<<1, 16>>>();
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Kernel execution completed successfully.\n");
    return 0;
}
