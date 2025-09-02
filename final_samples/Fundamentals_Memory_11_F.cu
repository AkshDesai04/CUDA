/*
Can you write to a __constant__ variable from within a kernel? Try it and observe the compiler error.
The goal of this program is to demonstrate that __constant__ memory on the device is read‑only from kernels. 
We declare a constant variable, attempt to modify it inside a __global__ function, and then launch that kernel. 
During compilation, NVCC will generate an error because assignments to __constant__ variables are not allowed inside kernels
(since constant memory is intended to be read only by device code). The error message typically looks like: 
"assignment to read-only variable". This code is intentionally incorrect to trigger that error.
*/

#include <cuda_runtime.h>
#include <stdio.h>

// Declare a constant memory variable
__constant__ int const_val = 0;

// Kernel attempting to write to the constant variable
__global__ void write_to_const()
{
    // This line should cause a compilation error
    const_val = 42;
}

int main()
{
    // Launch the kernel
    write_to_const<<<1, 1>>>();

    // Wait for the kernel to finish (for completeness)
    cudaDeviceSynchronize();

    printf("Kernel launched.\n");
    return 0;
}
