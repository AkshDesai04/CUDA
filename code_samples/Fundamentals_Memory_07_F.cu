```cpp
/*
Aim of the program:
Write a __host__ function that takes an int* and dereferences it. Call it with a device pointer. Observe the segfault.

Thinking process:
The goal is to illustrate that device memory addresses cannot be dereferenced from host code. 
We create a device memory allocation with cudaMalloc, write a value into that memory using a simple kernel,
and then pass the device pointer to a __host__ function that attempts to read from it using *ptr.
This will result in a segmentation fault or undefined behavior because the host cannot directly access device memory.
The program is kept minimal: a single kernel to set a value, a host function that dereferences, and the main routine that ties them together.
We include error checking only to the extent that it demonstrates the failure point.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that writes a value to the device pointer
__global__ void set_value(int* d_ptr, int val) {
    *d_ptr = val;
}

// Host function that attempts to dereference the pointer
__host__ void dereferenceDevicePtr(int* ptr) {
    // This dereference is illegal if ptr points to device memory
    printf("Dereferenced value: %d\n", *ptr);
}

int main() {
    int *d_ptr = nullptr;

    // Allocate memory on the device
    cudaError_t err = cudaMalloc((void**)&d_ptr, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel to set the value on the device
    set_value<<<1, 1>>>(d_ptr, 42);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_ptr);
        return 1;
    }

    // Call host function with the device pointer
    // This will likely cause a segmentation fault
    dereferenceDevicePtr(d_ptr);

    // Cleanup
    cudaFree(d_ptr);
    return 0;
}
```