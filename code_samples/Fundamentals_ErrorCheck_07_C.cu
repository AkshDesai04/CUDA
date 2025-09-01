```cpp
/*
How are synchronous errors reported? (Directly via the return code).

Thinking: 
In CUDA, certain API calls and kernel launches can fail synchronously, meaning
the failure is detected and reported immediately, allowing the program to
react without waiting for the GPU to finish. These synchronous errors are
reported directly via a return code of type `cudaError_t`. For example, a
kernel launch with an illegal grid or block configuration fails immediately
and can be detected by checking the return value of `cudaGetLastError()`.  
Other CUDA runtime API functions such as `cudaMalloc`, `cudaMemcpy`, or
`cudaDeviceSynchronize` also return a `cudaError_t` that indicates success
or failure. This program demonstrates synchronous error reporting in two
ways:
1.  A deliberately malformed kernel launch that fails immediately.
2.  An API call that fails because it requests more memory than is
    available, returning an error code directly.

The code prints the error code and a human‑readable message whenever a
synchronous error occurs, illustrating how to handle such errors in
production code. The program is self‑contained and can be compiled with
`nvcc -o sync_error_demo sync_error_demo.cu`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro to check CUDA error codes and print messages */
#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

/* Simple kernel that does nothing */
__global__ void do_nothing_kernel() {
    // No operation
}

int main(void) {
    /* -------------------------------------------------------------
     * 1. Synchronous error from an illegal kernel launch
     * ------------------------------------------------------------- */
    printf("Launching kernel with illegal configuration (grid.x = 0)...\n");
    /* Launching with grid.x = 0 is illegal and should fail synchronously. */
    do_nothing_kernel<<<0, 1>>>();
    /* Immediately check for launch errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Synchronous kernel launch error detected: %s\n",
               cudaGetErrorString(err));
    } else {
        printf("Kernel launch succeeded (unexpected).\n");
    }

    /* -------------------------------------------------------------
     * 2. Synchronous error from a CUDA runtime API call
     * ------------------------------------------------------------- */
    printf("\nAttempting to allocate an excessively large amount of memory...\n");
    /* Try to allocate 2^62 bytes (~4 EiB) which will definitely fail. */
    void* d_ptr = nullptr;
    size_t hugeSize = (size_t)1 << 62;
    err = cudaMalloc(&d_ptr, hugeSize);
    if (err != cudaSuccess) {
        printf("Synchronous cudaMalloc error detected: %s\n",
               cudaGetErrorString(err));
    } else {
        printf("cudaMalloc succeeded (unexpected).\n");
        /* Free memory if allocation somehow succeeded */
        CHECK_CUDA(cudaFree(d_ptr));
    }

    /* Clean exit */
    printf("\nProgram completed.\n");
    return 0;
}
```