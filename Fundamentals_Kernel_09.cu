/*
Create a `__device__` function called `square_me` that takes an integer `x` and returns `x*x`. Create a `__global__` kernel that calls this `square_me` function on its `threadIdx.x` and prints the result.

Thinking:
- The `square_me` function must be a `__device__` so it can run on the GPU.
- It will simply multiply the input by itself and return the result.
- The kernel should invoke this device function using the thread index `threadIdx.x`.
- We will use `printf` inside the kernel to print the thread index and the squared value.
- For the program to compile and run, we need to include the CUDA runtime header and the standard C `stdio.h` for `printf`.
- A simple `main` function will launch the kernel with a reasonable number of threads (e.g., 10) and synchronize.
- No complex error checking is required for this simple demonstration.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__device__ int square_me(int x) {
    return x * x;
}

__global__ void print_squares() {
    int val = square_me(threadIdx.x);
    printf("Thread %d: %d squared is %d\n", threadIdx.x, threadIdx.x, val);
}

int main() {
    int threadsPerBlock = 10;
    print_squares<<<1, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    return 0;
}
