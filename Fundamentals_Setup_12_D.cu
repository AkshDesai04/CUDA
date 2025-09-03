/*
Define a kernel that takes an `int*` argument, but try to launch it without providing any arguments.

Thinking:
- The goal is to show a kernel requiring an argument but launching with none.
- In CUDA, this will produce a compile-time error because the compiler checks the number of arguments.
- We'll define a simple kernel __global__ void foo(int* ptr).
- In main, allocate device memory, then call foo<<<1,1>>>() without passing d_ptr.
- The code compiles? No, it will fail to compile.
- The program demonstrates the compile-time mismatch.
- If one bypasses compile checks, a runtime error may occur.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void foo(int* ptr) {
    // Simple kernel that writes a value if ptr is not NULL
    if (ptr != NULL) {
        *ptr = 42;
    }
}

int main() {
    int *d_ptr = NULL;
    cudaMalloc((void**)&d_ptr, sizeof(int));

    // Attempt to launch the kernel without providing the required argument.
    // This line should cause a compile-time error: too few arguments to function 'foo'.
    foo<<<1, 1>>>(); // Missing argument

    cudaDeviceSynchronize();
    cudaFree(d_ptr);
    return 0;
}
