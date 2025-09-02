```cuda
/*
Aim: Try passing a pointer from `cudaMallocHost` to `free()` instead of `cudaFreeHost`. What happens?

Thinking:
- cudaMallocHost allocates page‑locked (pinned) host memory that is managed by the CUDA runtime.
- The proper deallocation routine is cudaFreeHost. Passing the returned pointer to the standard
  C library function free() is incorrect because free() expects memory allocated by malloc,
  calloc, or realloc.
- According to the CUDA programming guide, using free() on CUDA pinned memory results in
  undefined behaviour. On many systems this leads to a segmentation fault or corruption
  of the memory allocator's internal structures.
- To demonstrate, this program allocates pinned memory, writes to it, then calls free()
  on the pointer. It prints messages before and after the call to make it clear where
  the program stops executing. The program also prints the CUDA error state after the
  free() call (though free() will not set a CUDA error).
- The expected outcome when running this code is that it crashes (segmentation fault)
  or otherwise behaves unpredictably, confirming that free() cannot be used to deallocate
  CUDA pinned memory.

Note: Running this code on a system with a recent CUDA toolkit should exhibit a crash
or memory corruption. Do not use this pattern in production code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(void) {
    const size_t N = 10;
    int *pinned;
    cudaError_t err;

    // Allocate pinned memory
    err = cudaMallocHost((void**)&pinned, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    printf("Pinned memory allocated at %p\n", (void*)pinned);

    // Write to pinned memory
    for (size_t i = 0; i < N; ++i) {
        pinned[i] = (int)i;
    }
    printf("Pinned memory written.\n");

    // Incorrectly deallocate pinned memory using free()
    printf("Calling free() on pinned memory pointer %p...\n", (void*)pinned);
    free(pinned);  // <-- Undefined behaviour!

    // Attempt to print CUDA error (unlikely to change)
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after free(): %s\n", cudaGetErrorString(err));
    } else {
        printf("No CUDA error reported after free().\n");
    }

    // Try to access memory again to see if it crashed
    printf("Attempting to read pinned memory after free(): %d\n", pinned[0]);

    // Normally we would call cudaFreeHost here, but we already freed incorrectly.
    // So we skip it.

    printf("Program finished.\n");
    return EXIT_SUCCESS;
}
```