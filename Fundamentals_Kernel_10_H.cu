/*
Create a `__device__` function that takes the struct as an argument by value.

Thinking:
- The prompt requests a simple CUDA program that demonstrates a device function taking a struct argument by value.
- We'll define a basic struct `MyStruct` with a couple of members.
- The device function `processStruct` will accept a `MyStruct` by value, perform a trivial computation (e.g., increment fields), and print a message using `printf` (device-side printf requires `-rdc=true` and `-arch=sm_XX` during compilation).
- A kernel `kernelTest` will create an instance of `MyStruct`, pass it to `processStruct`, and also call it with another value to show that the function works.
- The host `main` will launch the kernel and synchronize.
- Include necessary headers and use `cudaMalloc`/`cudaFree` for any device memory if needed (none in this minimal example).
- Ensure the code is self-contained and compiles with nvcc.
*/

#include <stdio.h>
#include <cuda.h>

// Define a simple struct
struct MyStruct {
    int a;
    float b;
};

// Device function that takes MyStruct by value
__device__ void processStruct(MyStruct s) {
    // Modify the struct locally (does not affect the caller)
    s.a += 1;
    s.b += 1.0f;
    // Print the modified values
    printf("Device function received struct: a=%d, b=%f\n", s.a, s.b);
}

// Kernel that calls the device function
__global__ void kernelTest() {
    // Create a struct instance
    MyStruct local = {5, 3.14f};

    // Call the device function with the struct by value
    processStruct(local);

    // Another call with a literal struct
    processStruct((MyStruct){10, 2.71f});
}

int main() {
    // Launch the kernel
    kernelTest<<<1, 1>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Reset device (optional)
    cudaDeviceReset();

    return 0;
}
