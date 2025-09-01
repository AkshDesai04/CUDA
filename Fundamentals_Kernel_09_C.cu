/*
Aim: Explain the difference between `__global__` and `__device__` functions in a comment.

Thinking:
The user requested an explanation of the difference between `__global__` and `__device__` functions. 
To comply with the developer instruction, we must provide a .cu source file that contains a multiline comment at the very beginning. 
This comment should state the aim exactly as the user phrased it and also include the entire reasoning process I just described. 
After the comment, the actual code must be provided. The code should be a minimal but complete CUDA program that demonstrates the usage of a `__global__` kernel and a `__device__` helper function. 
We will create a small kernel that increments an integer on the device and prints the result from host. This demonstrates that `__global__` kernels can be launched from host and that `__device__` functions can be called from kernels. 
The program will:
1. Allocate memory on device.
2. Call a `__global__` kernel that uses a `__device__` function to perform the increment.
3. Copy the result back to host and print it.
The code will compile with `nvcc` and be self-contained. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Device helper function: increments a value by one
__device__ int add_one(int x) {
    return x + 1;
}

// Global kernel: calls the device helper and stores result in output
__global__ void kernel_add_one(int *out) {
    // Only one thread does the work for simplicity
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        *out = add_one(41); // 41 + 1 = 42
    }
}

int main(void) {
    int *d_out;
    int h_out = 0;

    // Allocate memory on the device
    cudaMalloc((void**)&d_out, sizeof(int));

    // Launch kernel: one block of one thread
    kernel_add_one<<<1, 1>>>(d_out);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Result from kernel: %d\n", h_out);

    // Free device memory
    cudaFree(d_out);

    return 0;
}
