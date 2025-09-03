/*
If a `cudaMalloc` fails, is the pointer it was supposed to write to modified? Or does it retain its previous value?

Thinking:
The question is about the side‑effects of a failed cudaMalloc call on the host pointer argument.  The CUDA API
specifies that cudaMalloc returns an error code and the pointer value passed in is *not* touched unless
the allocation succeeds.  To demonstrate this, we write a small CUDA C program that:

1. Sets a host pointer to a known dummy value.
2. Calls cudaMalloc with a size that is guaranteed to fail (a very large allocation).
3. Checks the return code; on failure we print the error and the current value of the pointer.
4. For comparison we also perform a successful allocation with a reasonable size, showing that the
   pointer is updated in that case.

The program is self‑contained, uses only the CUDA runtime API, and can be compiled with `nvcc`.
The output will show that after a failed cudaMalloc the pointer remains unchanged, confirming the
behaviour.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int *ptr = NULL;
    size_t large_size = (size_t)1 << 40; // 1 TB, likely to fail on most GPUs

    // Set ptr to a known dummy value
    ptr = (int*)0xDEADBEEF;

    printf("Before cudaMalloc (fail case), ptr = %p\n", (void*)ptr);

    // Attempt a huge allocation that will fail
    cudaError_t err = cudaMalloc((void**)&ptr, large_size);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed as expected: %s\n", cudaGetErrorString(err));
        printf("After failed cudaMalloc, ptr = %p (should be unchanged)\n", (void*)ptr);
    } else {
        // If by some miracle it succeeded, free the memory
        cudaFree(ptr);
    }

    // Now perform a successful allocation for comparison
    int *ptr_success = NULL;
    size_t small_size = 1024 * sizeof(int); // 4 KB, should succeed
    err = cudaMalloc((void**)&ptr_success, small_size);
    if (err == cudaSuccess) {
        printf("Successful cudaMalloc, ptr_success = %p\n", (void*)ptr_success);
        // Use the memory (optional) and free it
        cudaMemset(ptr_success, 0, small_size);
        cudaFree(ptr_success);
    } else {
        printf("Unexpected failure on small allocation: %s\n", cudaGetErrorString(err));
    }

    return 0;
}
