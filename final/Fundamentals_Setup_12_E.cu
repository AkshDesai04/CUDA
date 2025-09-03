/*
Try to declare a non-const static variable inside a kernel and observe the compilation error.

Thinking:
The task is to demonstrate that CUDA does not allow a non-const static variable to be declared inside a __global__ kernel. In C/C++ a static variable inside a function has function scope and persists across calls. On the GPU, a static variable inside a device function would imply shared or global persistence across threads, which CUDA does not support unless the variable is const or explicitly declared with __device__ or __constant__ qualifiers. Therefore, writing a kernel that declares `static int` should trigger a compilation error when building with nvcc. The code below includes a simple kernel that attempts this, along with a trivial host launch to make the compilation complete. When compiled, nvcc will produce an error such as "static variables inside __global__ function not allowed" or a similar message, confirming the restriction. This program is intentionally minimal to focus on the compilation error and avoid any runtime output issues.
*/
#include <stdio.h>

__global__ void testKernel()
{
    static int staticVar = 0; // This should cause a compilation error in CUDA
    printf("Static var: %d\n", staticVar);
}

int main()
{
    testKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}