/*
Aim: Use your macro on a call that is guaranteed to fail, like `cudaMalloc` for 100TB, to test its output.

Thinking:
1. Define a macro `CUDA_CHECK` that wraps any CUDA runtime API call, checks the returned `cudaError_t`,
   prints an informative message including file and line number if the call fails, and exits the program.
2. In `main`, attempt to allocate an enormous amount of device memory (100 TB) using `cudaMalloc`.
   This allocation will definitely fail on any realistic GPU, generating a `cudaErrorMemoryAllocation`.
3. Pass the `cudaMalloc` call through `CUDA_CHECK` so that the failure triggers the error handling
   path of the macro, allowing us to observe its output.
4. Include the necessary headers (`stdio.h`, `stdlib.h`, `cuda_runtime.h`) and compile with
   `nvcc test_fail.cu -o test_fail`.
5. When executed, the program should print a CUDA error message indicating the failure of the
   `cudaMalloc` call and terminate.

This program demonstrates the usefulness of a concise error-checking macro in CUDA C code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err);                                            \
        }                                                         \
    } while (0)

int main(void) {
    // Allocate an absurdly large amount of device memory (100 TB)
    size_t huge_size = 100ULL * 1024 * 1024 * 1024 * 1024ULL; // 100 TB
    void *dev_ptr = NULL;

    // This call is expected to fail; the macro will catch and report the error
    CUDA_CHECK(cudaMalloc(&dev_ptr, huge_size));

    // If, for some reason, the allocation succeeds (unlikely), free the memory
    CUDA_CHECK(cudaFree(dev_ptr));

    printf("Unexpectedly succeeded in allocating %zu bytes on the device.\n", huge_size);
    return 0;
}
