/*
 * Aim: Pass an integer literal (e.g., 5) as an argument to the empty kernel.
 *
 * Thinking:
 * The task requires writing a CUDA program that demonstrates passing an
 * integer literal to a kernel that has no functional body (i.e., an
 * "empty kernel").  The kernel must still accept an integer parameter, but
 * inside the kernel no operations need to be performed.  We will launch the
 * kernel with the literal value `5` to satisfy the requirement.
 *
 * Implementation details:
 * 1. Use `__global__ void emptyKernel(int value)` as the kernel.  The body
 *    of the kernel will be left empty (no statements).
 * 2. In `main`, launch the kernel with a 1×1 thread block configuration
 *    and pass the integer literal `5` directly in the kernel launch
 *    arguments.
 * 3. Include a simple error checking helper to ensure the kernel launch
 *    succeeds.
 * 4. Add `cudaDeviceSynchronize()` to make sure the kernel completes
 *    before the host exits.
 *
 * This simple program demonstrates how a kernel argument can be passed
 * directly as a literal, even if the kernel does not use it.  The code
 * is self‑contained and can be compiled with `nvcc` to produce an
 * executable that runs on a CUDA capable device.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Empty kernel that accepts an integer argument.
__global__ void emptyKernel(int value)
{
    // Intentionally left blank.
}

int main(void)
{
    // Launch the kernel with the integer literal 5.
    emptyKernel<<<1, 1>>>(5);

    // Ensure kernel execution completes.
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Kernel launched with integer literal 5.\n");
    return 0;
}
