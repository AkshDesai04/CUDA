/*
 * Aim: Demonstrate its use by launching a kernel with an invalid configuration (e.g., too many threads) and catching the error with `CHECK(cudaPeekAtLastError())`.
 *
 * Thinking:
 * 1. We want to trigger a CUDA launch error by providing an invalid block dimension that exceeds the device's limit for threads per block.
 * 2. The maximum number of threads per block on most GPUs is 1024 (may be 2048 on some newer GPUs, but 1024 is safe for demonstration).
 * 3. By setting blockDim.x to 2048 we guarantee an invalid configuration.
 * 4. After the kernel launch we call `cudaPeekAtLastError()` which returns the error status of the last launch.
 * 5. The `CHECK` macro will test the error code; if it is not `cudaSuccess` it prints the error message and exits.
 * 6. This demonstrates both the invalid configuration and the error-checking mechanism.
 * 7. The kernel itself does nothing; it's only a placeholder to trigger the launch.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check CUDA errors */
#define CHECK(call)                                                  \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d - %s (%d)\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err), err); \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

/* Dummy kernel that does nothing */
__global__ void dummyKernel()
{
    // Intentionally left blank
}

int main(void)
{
    /* Launch configuration: too many threads per block (e.g., 2048 > 1024 limit) */
    dim3 grid(1);
    dim3 block(2048);  // Invalid configuration

    /* Attempt to launch the kernel */
    dummyKernel<<<grid, block>>>();

    /* Catch the launch error */
    CHECK(cudaPeekAtLastError());

    /* If we reach here, something went wrong because CHECK should have exited */
    printf("Kernel launched successfully (unexpected).\n");

    return 0;
}
