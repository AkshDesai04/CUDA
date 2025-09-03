/*
Write a `__host__` function that takes an `int*` and dereferences it. Call it with a device pointer. Observe the segfault.

Thinking:
- The goal is to illustrate that a host function cannot safely dereference a pointer that points to device memory.
- We will allocate an integer on the GPU using cudaMalloc.
- We will write a simple kernel that writes a known value to that device memory so that if the host could read it, we would see the correct value.
- The host function, marked `__host__`, will take an `int*`, attempt to dereference it, and print the value.
- When we call this host function with the device pointer, the dereference will attempt to access GPU memory from the CPU side, which is not mapped into the CPU address space. This should cause a segmentation fault.
- The code will compile with `nvcc` and run; the user can observe the segfault in the output or debugger.
- No special error handling is added beyond a simple cudaGetLastError check for the kernel launch, because the focus is on the segfault.
- The code is written in standard C++ (which CUDA supports) and uses CUDA runtime API.
- The file ends with a `.cu` extension so it can be compiled by `nvcc`.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Host function that dereferences an int pointer and prints its value
__host__ void host_dereference(int *ptr) {
    // Attempt to read from the pointer (which we expect to be a device pointer)
    int val = *ptr; // This should cause a segmentation fault
    std::cout << "Dereferenced value: " << val << std::endl;
}

// Simple kernel to write a value into device memory
__global__ void set_value(int *ptr, int value) {
    *ptr = value;
}

int main() {
    int *d_ptr = nullptr;
    // Allocate memory on the device
    cudaError_t err = cudaMalloc((void**)&d_ptr, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // Launch kernel to set the value to 42
    set_value<<<1, 1>>>(d_ptr, 42);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_ptr);
        return EXIT_FAILURE;
    }

    // Ensure kernel execution completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_ptr);
        return EXIT_FAILURE;
    }

    std::cout << "Calling host_dereference with device pointer..." << std::endl;
    // This call should trigger a segmentation fault
    host_dereference(d_ptr);

    // Free device memory (unreachable if segfault occurs)
    cudaFree(d_ptr);
    return EXIT_SUCCESS;
}
