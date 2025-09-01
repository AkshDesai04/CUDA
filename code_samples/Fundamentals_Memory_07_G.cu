/*
Write a `__global__` function that takes an `int*` and dereferences it. Call it with a host pointer. Observe the error at runtime.

Thinking:
The goal is to demonstrate that passing a host pointer directly to a device kernel causes a runtime error. In CUDA, device kernels cannot access host memory directly; the pointer must reference device-allocated memory. To illustrate this, we create a simple global kernel that dereferences its `int*` argument and prints the value. In the host code, we declare an `int` on the stack, take its address, and pass that address to the kernel. When the kernel executes, it will attempt to read from the host memory location, which is invalid on the device. CUDA will detect this as an "invalid device pointer" error during kernel launch or when the kernel tries to access memory. We then capture the error with `cudaGetLastError()` and `cudaDeviceSynchronize()` and print it out to the console. This program will compile and run, but will produce a runtime error, clearly showing the mismatch between host and device memory spaces.

The code below includes:
- Necessary includes for CUDA and C I/O.
- A global kernel that dereferences the pointer.
- A main function that sets up a host int, launches the kernel with the host pointer, and checks for errors.
- Printing of the error messages so that the user can observe the runtime failure.

Compile with:
    nvcc -o host_ptr_error host_ptr_error.cu

Run with:
    ./host_ptr_error
The output will include the error messages indicating an invalid device pointer.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void derefKernel(int* ptr)
{
    // Attempt to dereference the passed pointer.
    // This should fail because ptr points to host memory.
    int val = *ptr;
    printf("Kernel read value: %d\n", val);
}

int main()
{
    // Host integer
    int hostInt = 42;
    int* hostPtr = &hostInt;

    // Launch kernel with host pointer (incorrect usage)
    derefKernel<<<1, 1>>>(hostPtr);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to catch any runtime errors during kernel execution
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
    }

    // Clean up
    cudaDeviceReset();

    return 0;
}
