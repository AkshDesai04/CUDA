/*
Deliberately mismatch the type of an argument, e.g., pass a `float` where the kernel expects an `int*`.

Thinking:
I want to show how a type mismatch between host and device arguments can lead to undefined
behaviour.  The kernel will be written to accept an `int*` for the output buffer, but on the
host side I will pass the address of a `float` variable (casted to `int*`).  The kernel will
write an integer result into the memory that holds the float, corrupting its bit pattern.
After kernel execution I will print the float to see the effect of writing an int into it.
This demonstrates a mismatch in argument types and how it can silently produce incorrect
results.  The code is deliberately simple and is intended for educational purposes only.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that writes the sum of two ints to the output location.
__global__ void addKernel(int *c, const int *a, const int *b)
{
    // Only one thread will be launched.
    c[0] = a[0] + b[0];
}

// Helper macro for checking CUDA errors.
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main(void)
{
    const int n = 1;
    int *d_a = NULL;
    int *d_b = NULL;
    // No device memory for the output; we will misuse a float on the host.

    // Allocate device memory for input arrays.
    CHECK_CUDA(cudaMalloc((void**)&d_a, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, n * sizeof(int)));

    // Host input values.
    int h_a = 3;
    int h_b = 4;

    // Copy inputs to device.
    CHECK_CUDA(cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice));

    // Host float that will incorrectly receive an int pointer.
    float f = 0.0f;

    // Cast the address of the float to int* to satisfy the kernel signature.
    int *miscastPtr = (int*)&f;

    // Launch the kernel with the mismatched argument.
    addKernel<<<1, 1>>>(miscastPtr, d_a, d_b);

    // Wait for kernel completion.
    CHECK_CUDA(cudaDeviceSynchronize());

    // Print the value of the float after kernel execution.
    // The bit pattern of the float now represents the integer 7 (3+4).
    printf("Value of float after kernel execution: %f\n", f);

    // Clean up device memory.
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    // Reset device (optional).
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
