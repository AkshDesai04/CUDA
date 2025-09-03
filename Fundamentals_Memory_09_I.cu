```cuda
/*
Aim of the program: Try passing a pointer from `cudaMallocHost` to `free()` instead of `cudaFreeHost`. What happens?

Thinking:
The CUDA runtime provides two sets of allocation functions:
  - cudaMallocHost() / cudaFreeHost() for page-locked (pinned) host memory.
  - cudaMalloc() / cudaFree() for device memory.
The pointer returned by cudaMallocHost() is allocated by the CUDA driver, not by the host C runtime.
If we call the standard C library function free() on that pointer, we are passing it to a deallocation routine that expects memory allocated by malloc() or similar.
This mismatch leads to undefined behavior: the runtime may not be able to correctly release the pinned memory, could corrupt the host heap, or crash the program.
We can demonstrate this by allocating pinned memory, filling it, then freeing it with free() and attempting to use it again.
The program may crash, print garbage, or seemingly work depending on the system, but it is incorrect.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const int N = 10;
    int *h_arr = NULL;

    // Allocate pinned host memory
    cudaError_t err = cudaMallocHost((void**)&h_arr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    printf("Pinned host memory allocated at %p\n", (void*)h_arr);

    // Initialize the array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i * 10;
    }
    printf("Array initialized.\n");

    // Incorrectly free pinned memory with free()
    free(h_arr);
    printf("Pinned memory freed with free().\n");

    // Attempt to read from the supposedly freed memory
    // This is undefined behavior; the program may crash or print garbage.
    printf("Trying to read first element after free: %d\n", h_arr[0]);

    // If the program hasn't crashed, attempt to free again with cudaFreeHost to see double free handling
    err = cudaFreeHost(h_arr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFreeHost failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("Pinned memory freed again with cudaFreeHost.\n");
    }

    return EXIT_SUCCESS;
}
```