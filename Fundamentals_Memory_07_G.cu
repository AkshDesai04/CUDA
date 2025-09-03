```cuda
/*
Write a `__global__` function that takes an `int*` and dereferences it. Call it with a host pointer. Observe the error at runtime.

Thought process:
1. A __global__ kernel runs on the GPU and expects pointers to device memory. Passing a host pointer (allocated with `malloc` or on the stack) will result in an illegal memory access because the GPU cannot dereference host addresses.
2. To illustrate this, we write a kernel that simply reads the integer pointed to by the argument and writes it to a device variable (or simply prints the value via printf).
3. In main, we allocate an int on the host, set a value, and then launch the kernel with that host pointer.
4. After the kernel launch we call `cudaGetLastError()` and `cudaDeviceSynchronize()` to capture any runtime errors. We print the error string if one occurs.
5. This code will compile successfully but will fail at runtime with an error like "invalid device pointer" or "invalid memory access" because the GPU tries to access a host memory address.
6. The example demonstrates the importance of allocating memory with cudaMalloc or using cudaMemcpy to copy data to device memory before passing pointers to kernels.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that dereferences an int pointer and prints the value
__global__ void derefKernel(int* ptr)
{
    // Attempt to read from the pointer
    int val = *ptr; // This will cause an illegal memory access if ptr is a host pointer
    // Print the value (optional, may not execute if illegal access occurs)
    printf("Kernel: value = %d\n", val);
}

int main()
{
    // Allocate an int on the host
    int hostInt = 42;

    // Launch kernel with host pointer (incorrect usage)
    derefKernel<<<1, 1>>>( &hostInt );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to catch any runtime errors from the kernel
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("Kernel executed successfully (unexpected).\n");
    }

    return 0;
}
```