```cpp
/*
In the host, allocate the struct using `cudaMallocManaged` instead of `cudaMalloc`. Initialize on host, launch kernel to modify, synchronize, and then access directly on host to verify.

Thinking:
The goal is to demonstrate the use of Unified Memory (`cudaMallocManaged`) to allocate a struct that can be accessed by both host and device code without explicit copy operations. The program will:
1. Define a simple struct `MyStruct` with a few fields.
2. Allocate a single instance of this struct on the GPU using `cudaMallocManaged`.
3. Initialize the struct on the host side.
4. Launch a kernel that modifies the struct's fields.
5. Synchronize the device to ensure the kernel has finished.
6. Read back and print the struct fields on the host to verify the modifications.
This showcases how Unified Memory simplifies memory management for small data structures that are accessed by both host and device.
*/
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

struct MyStruct {
    int a;
    float b;
    double c;
};

// Kernel that modifies the struct
__global__ void modifyStructKernel(MyStruct *s) {
    // Simple operations
    s->a += 10;
    s->b *= 2.0f;
    s->c -= 3.141592653589793;
}

int main() {
    MyStruct *d_struct;

    // Allocate unified memory
    cudaError_t err = cudaMallocManaged(&d_struct, sizeof(MyStruct));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate unified memory (error code %d)!\n", err);
        return -1;
    }

    // Initialize on host
    d_struct->a = 5;
    d_struct->b = 3.14f;
    d_struct->c = 1.61803398875;

    printf("Before kernel launch:\n");
    printf("  a = %d\n", d_struct->a);
    printf("  b = %f\n", d_struct->b);
    printf("  c = %lf\n", d_struct->c);

    // Launch kernel with one thread
    modifyStructKernel<<<1,1>>>(d_struct);

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        cudaFree(d_struct);
        return -1;
    }

    // Access directly on host
    printf("After kernel execution:\n");
    printf("  a = %d\n", d_struct->a);
    printf("  b = %f\n", d_struct->b);
    printf("  c = %lf\n", d_struct->c);

    // Clean up
    cudaFree(d_struct);
    return 0;
}
```