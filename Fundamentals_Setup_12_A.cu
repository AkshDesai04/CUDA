/*
Create a syntax error by misspelling `__global__` as `__globol__` and observe the compiler error.

Thinking:
- The goal is to demonstrate how the CUDA compiler reacts to a misspelled kernel launch attribute.
- We will write a minimal CUDA C program that attempts to declare a kernel using the incorrect attribute `__globol__`.
- The rest of the program will be syntactically correct so that the only error arises from the misspelling.
- A simple kernel that does nothing is enough, but we also include a tiny host main that allocates memory, launches the kernel, and cleans up.
- The compiler error should point out that `__globol__` is an unknown attribute or identifier.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Intentionally misspelled kernel launch attribute
__globol__ void myKernel(int* d_arr) {
    // Kernel body is intentionally empty
}

int main() {
    int *d_arr;
    cudaError_t err;

    err = cudaMalloc((void**)&d_arr, 10 * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch the kernel (this will not compile due to the misspelled attribute)
    myKernel<<<1, 10>>>(d_arr);

    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
