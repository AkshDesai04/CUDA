/*
Aim: Define a kernel that takes an int* argument, but try to launch it without providing any arguments.

Thinking:
- In CUDA, a kernel that declares a parameter must receive a matching argument at launch.
- The compile-time error is expected if the launch omits the argument.
- This program demonstrates that mismatch.
- The kernel will simply write a value to the pointer, but we purposely do not pass any argument.
- The code will not compile successfully; the compiler will report "too few arguments to function" or similar.
- This is intentional to illustrate the requirement.
*/

#include <cuda_runtime.h>
#include <iostream>

// Kernel that expects an int* argument
__global__ void myKernel(int* ptr) {
    // Attempt to dereference the pointer (unsafe if null)
    if (ptr) {
        *ptr = 42;
    }
}

int main() {
    // Allocate memory on device (not used due to missing argument launch)
    int* d_ptr = nullptr;
    cudaMalloc(&d_ptr, sizeof(int));

    // Intentionally launch kernel without providing the required argument
    // This line should cause a compile-time error: too few arguments to function
    myKernel<<<1, 1>>>();

    // Check for launch errors (will not be reached if compilation fails)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    // Free device memory
    cudaFree(d_ptr);
    return 0;
}
