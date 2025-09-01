/*
Aim of the program: Define a nested struct (a struct that contains another struct as a member). Pass it by value to a kernel and access a member of the inner struct.
Thinking:
- The program demonstrates how to define nested structs in C/C++ for CUDA, pass them by value to a kernel, and use the inner struct's members.
- We create an Inner struct with some members, then Outer struct containing an Inner instance and another member.
- In the kernel, we receive an Outer parameter by value; the CUDA compiler copies the entire struct onto the device stack. Inside, we access o.inner.a and o.c to compute a result and store in device output array.
- On the host, we prepare an Outer instance, allocate device output array, launch kernel with single thread, copy back result, and print it.
- This shows that nested structs are supported as kernel arguments and accessible in device code.
*/

#include <cstdio>
#include <cuda_runtime.h>

struct Inner {
    int a;
    float b;
};

struct Outer {
    Inner inner;
    int c;
};

__global__ void nestedStructKernel(Outer o, int* out) {
    // Access a member of the inner struct
    int val = o.inner.a + o.c;
    out[threadIdx.x] = val;
}

int main() {
    // Prepare host data
    Outer h_outer;
    h_outer.inner.a = 5;
    h_outer.inner.b = 3.14f;
    h_outer.c = 10;

    int h_result = 0;
    int* d_result = nullptr;

    // Allocate device memory
    cudaMalloc((void**)&d_result, sizeof(int));

    // Launch kernel with 1 block and 1 thread
    nestedStructKernel<<<1, 1>>>(h_outer, d_result);

    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Result from kernel: %d\n", h_result);  // Expected 15

    // Clean up
    cudaFree(d_result);

    return 0;
}
