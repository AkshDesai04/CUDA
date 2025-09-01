/*
Aim of the program: Pass a string (char*) from the host to the kernel and try to print it. Observe what happens and why it doesn't work as expected.

Thinking:
When we pass a raw host pointer (char*) to a CUDA kernel, we are simply copying the pointer value (the address) to the device side. The device cannot directly access host memory, so dereferencing that pointer in device code will either produce garbage, a crash, or nothing at all. The correct approach is to allocate device memory for the string, copy the string contents from host to device, and then pass the device pointer to the kernel. This example intentionally uses the wrong method to illustrate the problem. The kernel will try to print the string using device printf, which will attempt to dereference the host pointer and will fail, usually resulting in no output or undefined behavior. The program also demonstrates how to check for CUDA errors and use cudaDeviceSynchronize() to flush device output.

The key points to observe:
1. A pointer value from host memory is not valid on the device.
2. Device code cannot read from host memory without an explicit copy.
3. Using device printf on an invalid pointer will not produce the expected string.
4. To correctly pass a string, we must allocate device memory and copy the string contents.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple CUDA error checking macro */
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* Kernel that attempts to print a string passed from the host */
__global__ void printStringKernel(char *str)
{
    // Print the pointer value for debugging
    printf("[Kernel] Pointer value received: %p\n", (void*)str);

    // Attempt to print the string contents
    // This will likely fail because 'str' points to host memory
    printf("[Kernel] Attempting to print string: %s\n", str);
}

int main(void)
{
    /* Host string */
    const char *h_str = "Hello from host";

    /* Device pointer that will hold the copied host pointer value */
    char *d_str = NULL;

    /* Copy the host pointer value to device memory */
    CUDA_CHECK(cudaMemcpy(&d_str, &h_str, sizeof(char*), cudaMemcpyHostToDevice));

    /* Launch kernel with a single thread to perform the printing */
    printStringKernel<<<1, 1>>>(d_str);

    /* Wait for kernel to finish and flush device printf output */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Clean up (not strictly necessary for this simple demo) */
    CUDA_CHECK(cudaFree(d_str));

    printf("[Host] Program completed.\n");
    return 0;
}
