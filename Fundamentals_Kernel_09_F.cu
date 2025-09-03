```cuda
/*
Aim of the program:
Place your __device__ function definition after the __global__ kernel that calls it. Does it compile? (It might require a forward declaration).

Thinking:
In CUDA, a device function must be declared before it is used by a kernel. If we place the definition after the kernel, the compiler will not yet know about it when parsing the kernel. This can be fixed by providing a forward declaration (prototype) of the device function before the kernel definition. The program below demonstrates this: a simple kernel that calls a device function to add two integers. The device function is defined after the kernel, and a forward declaration is provided. The program should compile and run correctly, printing the result of the addition.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Forward declaration of the device function
__device__ int add(int a, int b);

// Kernel that calls the device function
__global__ void addKernel(int *result, int a, int b) {
    // Each thread computes the sum of a and b
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) { // Only thread 0 writes to result
        *result = add(a, b);
    }
}

// Definition of the device function (after the kernel)
__device__ int add(int a, int b) {
    return a + b;
}

int main() {
    const int a = 3;
    const int b = 5;
    int h_result = 0;
    int *d_result;

    // Allocate device memory
    cudaMalloc((void**)&d_result, sizeof(int));

    // Launch kernel (1 block, 1 thread)
    addKernel<<<1, 1>>>(d_result, a, b);

    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Result of %d + %d = %d\n", a, b, h_result);

    // Clean up
    cudaFree(d_result);

    return 0;
}
```