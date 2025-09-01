/*
Create a `__device__` function called `square_me` that takes an integer `x` and returns `x*x`. Create a `__global__` kernel that calls this `square_me` function on its `threadIdx.x` and prints the result.
I first considered how to structure the CUDA code so that the device function `square_me` is simple and compiles correctly. The function only needs to take an `int` and return the square, so it can be inlined easily. The kernel will then call this function with the thread’s x‑index and use `printf` to output the result. Since `printf` inside a kernel is supported on compute capabilities 2.0 and higher, this will work on most recent GPUs. I added a small `main` that launches the kernel with a single block and a few threads, then synchronizes to ensure the output is flushed before the program exits. No additional error handling or advanced features were required for this straightforward demonstration. 
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
    int threads = 10;                     // Number of threads in the block
    int blocks = 1;                       // One block is sufficient for this demo

    // Launch the kernel
    square_kernel<<<blocks, threads>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    return 0;
}
