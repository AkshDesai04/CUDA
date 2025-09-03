```cuda
/*
Aim of the program:
Create a __device__ function called square_me that takes an integer x and returns x*x. 
Create a __global__ kernel that calls this square_me function on its threadIdx.x and prints the result.

Thinking process:
1. Define the __device__ function square_me that simply returns the square of its input.
2. Create a __global__ kernel that will be launched with some number of threads. 
   Each thread will call square_me with its own threadIdx.x and then print the result using device-side printf. 
   The printf format string will display the thread index, the input value, and the squared result.
3. In the host main function, launch the kernel with a single block and a modest number of threads (e.g., 8) to keep output readable.
   After launching, synchronize the device to ensure all device-side printf outputs are flushed before the program exits.
4. Use standard <stdio.h> for printf and include minimal error checking for the kernel launch. 
   This code requires a compute capability of at least 2.0 to support device printf. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

__device__ int square_me(int x) {
    return x * x;
}

__global__ void square_kernel() {
    int val = square_me(threadIdx.x);
    printf("Thread %d: %d squared is %d\n", threadIdx.x, threadIdx.x, val);
}

int main() {
    // Launch kernel with 1 block of 8 threads
    square_kernel<<<1, 8>>>();
    // Wait for kernel to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
```