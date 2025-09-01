/*
Aim: Deliberately mismatch the type of an argument, e.g., pass a `float` where the kernel expects an `int*`.

Thinking: The goal is to illustrate a compile-time error due to type mismatch. 
We declare a simple CUDA kernel that takes an int pointer, allocate a float on the host, 
and try to launch the kernel with the float variable as an argument. 
This will trigger a compilation error because the kernel parameter type does not match the argument type.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void myKernel(int* d_array) {
    // kernel body not relevant
    int idx = threadIdx.x;
    d_array[idx] = idx;
}

int main() {
    float hostFloat = 3.14f;

    // Intentionally passing a float where an int* is expected.
    myKernel<<<1,1>>>(hostFloat); // This line will cause a compile-time type mismatch error.

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    return 0;
}
