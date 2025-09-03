```cuda
/*
Create a __device__ function add_two(int a, int b) that returns a+b. Call it from your kernel.

Thinking:
- The goal is to demonstrate a simple device function that performs addition.
- We'll define `__device__ int add_two(int a, int b)` returning `a + b`.
- The kernel will invoke this device function, store the result in global memory,
  and also use `printf` (device printf) to show the computation.
- On the host side, we allocate device memory for the result, launch the kernel,
  copy the result back to the host, and print it.
- The code uses standard CUDA runtime API: `cudaMalloc`, `cudaMemcpy`, `cudaFree`.
- We'll include `<stdio.h>` for printing.
- The kernel will be a very small example that calls `add_two` with sample arguments.
- This code can be compiled with `nvcc` and run on a CUDA-capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Device function that adds two integers
__device__ int add_two(int a, int b) {
    return a + b;
}

// Kernel that calls the device function
__global__ void add_kernel(int *d_result) {
    // Example values
    int a = 3;
    int b = 5;
    // Call the device function
    int sum = add_two(a, b);
    // Store result in global memory
    *d_result = sum;
    // Optional: print from device
    printf("add_two(%d, %d) = %d\n", a, b, sum);
}

int main() {
    int h_result = 0;
    int *d_result = NULL;

    // Allocate device memory
    cudaMalloc((void **)&d_result, sizeof(int));

    // Launch kernel with 1 block and 1 thread
    add_kernel<<<1, 1>>>(d_result);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result on host
    printf("Result from device: %d\n", h_result);

    // Free device memory
    cudaFree(d_result);

    return 0;
}
```