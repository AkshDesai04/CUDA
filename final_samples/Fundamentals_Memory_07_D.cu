```cpp
/*
Aim of the program (as passed in the prompt word to word):
"If you print the value of a device pointer (`printf("%p", d_ptr);`), it prints an address. Explain that this address is only valid from the GPU's perspective."

Thinking:
1. A device pointer points to memory in the GPU's address space, which is separate from the host's address space.
2. When we print a device pointer inside a CUDA kernel using `printf`, the printed value is a GPU virtual address.
3. This address is meaningful only to the GPU; the host cannot dereference or use it directly because the host's memory mapping is different.
4. To illustrate this, we allocate memory on the device, print the pointer from both the host and from a kernel, and add comments explaining the difference.
5. The code will use `cudaMalloc` to allocate device memory, a simple kernel to print the device pointer, and host code to print the same pointer.
6. We also note that the host and device addresses are often completely unrelated, especially when using unified memory or different architectures.

The program demonstrates the concept and includes explanatory comments inside the code.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that prints the value of the device pointer
__global__ void printDevicePointer(const int *d_ptr)
{
    // The %p format specifier prints the pointer value as a hex address
    // This address is a GPU virtual address and is only valid from the GPU's perspective
    printf("Kernel sees device pointer at: %p\n", (const void *)d_ptr);
}

int main()
{
    const size_t N = 10;
    int *d_ptr = nullptr;

    // Allocate memory on the GPU
    cudaError_t err = cudaMalloc((void **)&d_ptr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Host prints the device pointer
    // This address is a host-side representation of the GPU pointer; it cannot be dereferenced by the host
    printf("Host sees device pointer at: %p\n", (const void *)d_ptr);

    // Launch a kernel that prints the device pointer
    printDevicePointer<<<1, 1>>>(d_ptr);

    // Wait for kernel to finish and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_ptr);
        return 1;
    }

    // Free the allocated device memory
    cudaFree(d_ptr);

    return 0;
}
```