/*
Aim of the program: Write a program that tries to access a device pointer from host code (e.g., `int* d_ptr; cudaMalloc(&d_ptr, ...); printf("%d", d_ptr[0]);`). Observe the segmentation fault or error this causes and explain why it happens.

Thinking:
- The program demonstrates a common mistake in CUDA programming: attempting to dereference a device pointer directly from host code.
- `cudaMalloc` allocates memory on the GPU and returns a pointer that is only valid in the device address space. The host cannot directly dereference this pointer; doing so will lead to undefined behavior.
- On most systems, the device address space is not mapped into the host process's virtual address space. Therefore, when the host tries to read `d_ptr[0]`, it will either cause a segmentation fault or, if the runtime library catches it, produce an error message such as "invalid device pointer".
- The program intentionally performs the incorrect dereference to illustrate the issue. After printing the erroneous access, it cleans up the allocated memory.
- Note: In a real-world scenario, you would use `cudaMemcpy` or device kernels to transfer data between host and device, not direct pointer dereferencing.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int *d_ptr;
    size_t N = 10;
    cudaError_t err = cudaMalloc((void**)&d_ptr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Attempting to read from a device pointer in host code...\n");

    // Incorrect: dereferencing a device pointer on the host.
    // This should cause a segmentation fault or a CUDA runtime error.
    int val = d_ptr[0];
    printf("Read value: %d\n", val);

    cudaFree(d_ptr);
    return 0;
}
