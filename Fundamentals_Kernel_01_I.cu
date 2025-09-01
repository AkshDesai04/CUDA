/*
Pass a float literal (e.g., 3.14f) as an argument to the empty kernel.
My thinking:
- The goal is to demonstrate passing a float literal to a CUDA kernel.
- We'll create a minimal kernel that accepts a float argument but does nothing.
- On the host side, we'll launch this kernel with a 3.14f literal.
- We'll use a simple launch configuration (1 block, 1 thread).
- Include basic error checking for CUDA API calls.
- The program will compile with nvcc and run, though no visible output is produced.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for simple CUDA error checking
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            return EXIT_FAILURE;                                        \
        }                                                               \
    } while (0)

// Empty kernel that takes a float argument
__global__ void emptyKernel(float val) {
    // Intentionally left empty
}

int main(void) {
    // Launch the kernel with 1 block and 1 thread, passing 3.14f
    emptyKernel<<<1, 1>>>(3.14f);

    // Synchronize to catch any launch errors
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Kernel launched with float literal 3.14f.\n");
    return EXIT_SUCCESS;
}
