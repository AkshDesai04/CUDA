```cuda
/*
Aim: Write a program that tries to access a device pointer from host code (e.g., `int* d_ptr; cudaMalloc(&d_ptr, ...); printf("%d", d_ptr[0]);`). Observe the segmentation fault or error this causes and explain why it happens.

Thinking:
- The program allocates an integer on the GPU using cudaMalloc, which returns a pointer that is only valid in the device address space.
- In host code, the pointer variable d_ptr holds the device address, but the host CPU cannot directly dereference it because the address is not mapped into the host's virtual memory.
- When the host attempts to read d_ptr[0] via printf, the CPU tries to access an invalid memory location, resulting in a segmentation fault or an illegal memory access error.
- The program will print a stack trace or abort, illustrating that device pointers must not be dereferenced on the host; instead, data should be transferred with cudaMemcpy or accessed within a kernel.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int *d_ptr = NULL;
    size_t size = sizeof(int);

    cudaError_t err = cudaMalloc(&d_ptr, size);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Attempt to dereference device pointer on host
    printf("Attempting to read d_ptr[0] on host: ");
    // This line will cause a segmentation fault or illegal memory access
    printf("%d\n", d_ptr[0]);

    // Clean up
    cudaFree(d_ptr);
    cudaDeviceReset();

    return 0;
}
```