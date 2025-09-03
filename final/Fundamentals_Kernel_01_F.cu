/*
Aim: Create a second empty kernel with a different name and launch both kernels from the host.

Thoughts:
- The program must be written in CUDA C and compiled as a .cu file.
- The first part of the file must be a multi‑line comment.  The comment
  should contain the exact phrase that was passed in the prompt – “Create a second empty kernel with a different name and launch both kernels from the host.”
  In addition to the aim, I will add a detailed explanation of how the code is structured.
- Two empty kernels are required.  The simplest way is to declare them as
  `__global__ void kernel1(void) {}` and `__global__ void kernel2(void) {}`.
  No device memory or logic is needed inside them.
- The host code will:
  1. Launch `kernel1` with a minimal grid and block size (e.g. `<<<1,1>>>`).
  2. Check for launch errors using `cudaGetLastError()` and `cudaDeviceSynchronize()`.
  3. Launch `kernel2` in the same way.
  4. Again check for errors.
  5. Print a simple success message and exit.
- Error handling is added to make the program robust.  If any error occurs,
  it is printed to `stderr` and the program terminates with a non‑zero exit code.
- The program includes standard headers `<stdio.h>` and `<cuda_runtime.h>`.
- `main` returns 0 on success.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Empty kernel #1 */
__global__ void kernel1(void) {
    // No operation
}

/* Empty kernel #2 */
__global__ void kernel2(void) {
    // No operation
}

/* Helper macro for error checking */
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void) {
    /* Launch kernel1 */
    kernel1<<<1, 1>>>();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("kernel1 launched successfully.\n");

    /* Launch kernel2 */
    kernel2<<<1, 1>>>();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("kernel2 launched successfully.\n");

    printf("Both kernels executed successfully.\n");
    return 0;
}
