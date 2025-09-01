```cpp
/*
Aim of the program:
If you print the value of a device pointer (printf("%p", d_ptr);), it prints an address.
Explain that this address is only valid from the GPU's perspective.

Thought process:
The goal is to demonstrate that a pointer obtained via cudaMalloc resides in the device's
address space. When we print that pointer on the host, the printed value is meaningless to
the host CPU because the host and device have separate memory address spaces (in unified
memory architectures the addresses may coincide, but in general they do not). By printing
the pointer inside a kernel we can see the address that the GPU sees, and by printing the
same pointer value on the host we can see the address the host sees. The two values may
differ or may be the same, but the important point is that the host cannot dereference a
device pointer directly. The kernel can safely use the pointer, but the host must use
cudaMemcpy or other API calls to transfer data. This code allocates an array on the
device, launches a kernel that prints the device pointer, and then prints the same
pointer value from the host, illustrating the difference.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that prints the device pointer address
__global__ void printDevicePtrKernel(int *d_ptr)
{
    // Use CUDA printf from the device
    printf("Device side: d_ptr = %p\n", (void *)d_ptr);
}

// Helper macro for checking CUDA errors
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void)
{
    int *d_arr = nullptr;
    size_t n = 10;

    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc((void **)&d_arr, n * sizeof(int)));

    // Initialize device memory (optional)
    CUDA_CHECK(cudaMemset(d_arr, 0, n * sizeof(int)));

    // Launch kernel to print device pointer
    printDevicePtrKernel<<<1, 1>>>(d_arr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Print the same pointer value from the host
    printf("Host side: d_arr = %p\n", (void *)d_arr);

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));

    return 0;
}
```