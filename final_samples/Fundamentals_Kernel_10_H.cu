/*
Create a `__device__` function that takes the struct as an argument by value.

Thinking:
- Define a simple struct MyStruct with some fields.
- Write a __device__ function that accepts MyStruct by value.
- For demonstration, have the function print the struct's fields using device printf.
- Provide a kernel that constructs a MyStruct instance and calls the device function.
- In main, launch the kernel and synchronize.
- Include necessary headers and basic error checking.
- Ensure the code compiles as a .cu file and uses CUDA runtime API.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple struct definition
struct MyStruct {
    int a;
    float b;
};

// __device__ function that takes the struct by value
__device__ void processStruct(MyStruct s) {
    // Using device printf to output the values
    printf("In device function: a = %d, b = %f\n", s.a, s.b);
}

// Kernel that calls the __device__ function
__global__ void kernel() {
    // Create an instance of MyStruct
    MyStruct s = {42, 3.14f};
    // Call the device function with the struct by value
    processStruct(s);
}

int main() {
    // Launch the kernel with one block and one thread
    kernel<<<1, 1>>>();
    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
