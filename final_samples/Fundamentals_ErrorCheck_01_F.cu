/*
Aim: Use your macro on a call that is guaranteed to fail, like `cudaMalloc` for 100TB, to test its output.

Thinking:
The goal is to create a self‑contained CUDA program that demonstrates the behaviour of an error‑checking macro
when invoked on a CUDA runtime call that will inevitably fail.  
The macro should:

1. Execute a CUDA runtime call.
2. Inspect the returned `cudaError_t`.
3. If the call was not successful, print a helpful error message that includes:
   - The file name
   - The line number
   - The function name
   - The human‑readable error string returned by `cudaGetErrorString`.
4. Terminate the program so that no further CUDA operations are attempted.

For the test we will request an absurd amount of device memory – 100 TB – via `cudaMalloc`.  
On any typical GPU this allocation will exceed the available device memory and `cudaMalloc` will return `cudaErrorMemoryAllocation`.  
By wrapping this call with our macro we can observe the printed diagnostics.

Implementation details:
- Use `size_t` for the allocation size and calculate 100 TB with unsigned 64‑bit constants to avoid overflow.
- The macro uses a `do { … } while(0)` wrapper to be safe in all contexts.
- We include `<stdio.h>`, `<stdlib.h>`, and `<cuda_runtime.h>`.
- In `main` we declare a `void*` pointer, call the macro on `cudaMalloc`, and then exit successfully if it ever succeeds (which it should not).
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check the result of a CUDA runtime call.
 * If the call fails, prints file, line, and error string, then exits.
 */
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in file '%s', line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main(void) {
    /* Allocate 100 TB of device memory – guaranteed to fail on any real GPU. */
    size_t huge_size = 100ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL; // 100 * 2^40 bytes
    void *dev_ptr = NULL;

    /* This call should fail and trigger the macro's error handling. */
    CUDA_CHECK(cudaMalloc(&dev_ptr, huge_size));

    /* If we reach this point, the allocation unexpectedly succeeded. */
    printf("Unexpectedly succeeded in allocating %zu bytes on device.\n", huge_size);
    /* Clean up (unlikely to be executed). */
    cudaFree(dev_ptr);
    return 0;
}
