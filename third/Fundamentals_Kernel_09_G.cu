```c
/*
Create a `__device__` function that is recursive, like a factorial function. Try calling it from a kernel. Be aware of stack limitations.

Thinking:
- CUDA supports device-side recursion when compiled with relocatable device code (-rdc=true) and appropriate architecture.
- A simple factorial recursion is straightforward: base case n <= 1 returns 1, otherwise return n * factorial(n-1).
- Recursion depth is limited by the device stack size, which is typically small (few kilobytes). For safety, keep the recursion depth small (e.g., up to 20).
- We'll create a kernel that calls the recursive function for small n values (0-9) to avoid stack overflow.
- The host will set up data, launch the kernel, copy results back, and print them.
- No external libraries needed; use standard CUDA headers.
- Error checking is minimal for clarity but can be expanded.
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Recursive device factorial function
__device__ unsigned long long device_factorial(unsigned int n) {
    if (n <= 1) return 1ULL;
    return n * device_factorial(n - 1);
}

// Kernel that computes factorial for each input element
__global__ void factorial_kernel(const unsigned int* d_n, unsigned long long* d_result, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_result[idx] = device_factorial(d_n[idx]);
    }
}

int main() {
    const int count = 10;
    unsigned int h_n[count];
    unsigned long long h_result[count];

    // Prepare input: 0 to 9
    for (int i = 0; i < count; ++i) {
        h_n[i] = i;
    }

    // Device pointers
    unsigned int* d_n = nullptr;
    unsigned long long* d_result = nullptr;

    // Allocate device memory
    cudaMalloc((void**)&d_n, count * sizeof(unsigned int));
    cudaMalloc((void**)&d_result, count * sizeof(unsigned long long));

    // Copy input to device
    cudaMemcpy(d_n, h_n, count * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Launch kernel: one block of 'count' threads
    factorial_kernel<<<1, count>>>(d_n, d_result, count);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_result, d_result, count * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < count; ++i) {
        printf("factorial(%u) = %llu\n", h_n[i], h_result[i]);
    }

    // Clean up
    cudaFree(d_n);
    cudaFree(d_result);

    return 0;
}
```