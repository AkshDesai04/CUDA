/*
Aim of the program: Pass a float literal (e.g., 3.14f) as an argument to the empty kernel.

Thinking process:
1. The user wants a CUDA program that demonstrates how to pass a float literal to an empty kernel.
2. We'll create a kernel `__global__ void emptyKernel(float val)` that accepts a float parameter but does nothing inside.
3. In `main`, we will launch this kernel with a single thread (`<<<1, 1>>>`) and pass the literal `3.14f` as the argument.
4. To ensure the kernel execution completes before the program exits, we call `cudaDeviceSynchronize()` after the launch.
5. We'll include minimal error checking: after each CUDA call, we check the return status and print an error message if something goes wrong.
6. The file will be a valid `.cu` file that can be compiled with `nvcc`.

The code below follows these steps and compiles to a CUDA executable that demonstrates passing a float literal to a kernel.
*/

#include <cuda_runtime.h>
#include <stdio.h>

// Empty kernel that accepts a float argument
__global__ void emptyKernel(float val) {
    // The kernel intentionally does nothing with the argument.
    // It simply receives the value to demonstrate passing a float literal.
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",             \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                          \
        }                                                                 \
    } while (0)

int main(void) {
    // Launch the empty kernel with a float literal argument 3.14f
    CUDA_CHECK(emptyKernel<<<1, 1>>>(3.14f));

    // Wait for the kernel to finish execution
    CUDA_CHECK(cudaDeviceSynchronize());

    // Optional: print confirmation that kernel launched successfully
    printf("Kernel launched with float literal 3.14f.\n");

    return 0;
}
