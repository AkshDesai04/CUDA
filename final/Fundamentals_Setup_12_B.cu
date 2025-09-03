/*
Create an error by calling a regular C++ function (that is not marked `__device__`) from within your kernel.

Thinking:
- To force a compilation error in CUDA, we can attempt to call a host function from a device kernel.
- In CUDA, any function that is called from device code must be marked with `__device__` (or `__global__` for kernels).
- Defining a normal C++ function without `__device__` and calling it from a `__global__` kernel will cause the compiler to emit an error, as it will not find a device version of the function.
- The rest of the program is minimal: a dummy host function `foo()`, a kernel that calls `foo()`, and a `main()` that launches the kernel.
- The code will compile with `nvcc` but fail with a compile-time error indicating that `foo` is not defined for device usage.
*/

#include <iostream>

void foo() {
    // Regular host function, not marked __device__
    std::cout << "Hello from host function!" << std::endl;
}

__global__ void myKernel() {
    // Attempt to call the host function from device code
    foo(); // This will cause a compilation error
}

int main() {
    // Launch the kernel (though it will fail to compile)
    myKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "Kernel launched." << std::endl;
    return 0;
}
