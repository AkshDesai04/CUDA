/*
Aim of the program:
Create a `__device__` function that takes the struct as an argument by value.

Thinking:
To satisfy the requirement I first define a simple struct `MyStruct` containing two members (an `int` and a `float`) to demonstrate that a struct can be passed by value to a device function.  
The device function `myDeviceFunc` receives an instance of `MyStruct` by value, and for visibility it prints the contents of the struct using `printf`.  Printing is allowed in device code for devices of compute capability >= 2.0, which is commonly supported.

I then create a kernel `kernel` that accepts a `MyStruct` parameter (also by value) and calls `myDeviceFunc`.  
In `main` I allocate and initialize a `MyStruct` instance on the host, launch the kernel with one thread, and synchronize. The program uses standard CUDA runtime API calls: `cudaMallocManaged` for unified memory (easier to pass struct to kernel), `cudaDeviceSynchronize`, and error checking. The kernel launch configuration is a single block of a single thread as the purpose is to demonstrate passing by value, not parallelism.

The program is self‑contained, includes all necessary headers, and compiles with `nvcc`.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Define a simple struct with two members.
struct MyStruct {
    int a;
    float b;
};

// Device function that takes the struct by value.
__device__ void myDeviceFunc(MyStruct s) {
    // Print the contents of the struct.  This demonstrates that
    // the struct was copied into the function's parameter space.
    printf("Inside device: a = %d, b = %f\n", s.a, s.b);
}

// Kernel that receives the struct by value and calls the device function.
__global__ void kernel(MyStruct s) {
    myDeviceFunc(s);
}

// Helper macro for error checking.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main(void) {
    // Allocate unified memory for the struct and initialize it on the host.
    MyStruct *d_s;
    CUDA_CHECK(cudaMallocManaged(&d_s, sizeof(MyStruct)));
    d_s->a = 42;
    d_s->b = 3.14f;

    // Launch the kernel with one block of one thread.
    kernel<<<1, 1>>>(*d_s);

    // Wait for the kernel to finish.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up.
    CUDA_CHECK(cudaFree(d_s));

    return 0;
}
