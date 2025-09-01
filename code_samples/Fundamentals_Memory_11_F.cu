/*
Aim: Can you write to a __constant__ variable from within a kernel? Try it and observe the compiler error.

Thinking:
- In CUDA, __constant__ memory is designated as read‑only for device code. 
- The host can initialize it via cudaMemcpyToSymbol or at compile time, but kernels cannot modify it.
- Attempting an assignment to a __constant__ variable inside a kernel should trigger a compilation error from nvcc.
- To demonstrate this, we declare a __constant__ int, then write a kernel that tries to assign a value to it.
- The main function launches the kernel and synchronizes; the compilation error will occur before runtime.
- The error message will be something like "invalid assignment to const variable" or "cannot assign to read-only variable".
*/

#include <stdio.h>

__constant__ int constVal = 0;

__global__ void writeToConstant()
{
    // Attempt to modify constant memory from a kernel (should produce a compile-time error)
    constVal = 42;
}

int main()
{
    writeToConstant<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
