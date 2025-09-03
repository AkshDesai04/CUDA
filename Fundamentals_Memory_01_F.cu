```cuda
/* 
Aim: Try to `cudaMalloc` zero bytes. Does it return an error or a valid pointer?

This program tests the behavior of the CUDA runtime when allocating
zero bytes with `cudaMalloc`. According to the CUDA Runtime API,
`cudaMalloc` should return `cudaSuccess` even when the requested size
is zero, but the semantics of the returned pointer are not explicitly
defined. Some implementations return a non‑null pointer that cannot
be dereferenced, while others may return a null pointer. The purpose
of this test is to observe the returned error code, the value of the
pointer, and whether `cudaFree` accepts it without causing a crash.

The program performs the following steps:
1. Calls `cudaMalloc` with a size of 0 and stores the result in a
   `void*` pointer.
2. Prints the returned error code and the string representation
   of the error.
3. Prints the pointer value and indicates whether it is NULL.
4. Frees the pointer (if non‑NULL) to avoid memory leaks.
5. Exits with status 0.

Compile with: 
    nvcc -o cuda_zero_malloc cuda_zero_malloc.cu

Run with:
    ./cuda_zero_malloc
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    void *devPtr = NULL;
    size_t size = 0;

    // Attempt to allocate zero bytes on the device
    cudaError_t err = cudaMalloc(&devPtr, size);

    // Print the error code and message
    printf("cudaMalloc(%zu) returned error code: %d (%s)\n",
           size, err, cudaGetErrorString(err));

    // Print the pointer value
    printf("Returned pointer: %p\n", devPtr);
    if (devPtr == NULL) {
        printf("Pointer is NULL.\n");
    } else {
        printf("Pointer is NOT NULL.\n");
    }

    // Free the allocated memory if the pointer is non-NULL
    if (devPtr != NULL) {
        cudaError_t freeErr = cudaFree(devPtr);
        printf("cudaFree returned error code: %d (%s)\n",
               freeErr, cudaGetErrorString(freeErr));
    } else {
        // Even if NULL, cudaFree is safe to call
        cudaError_t freeErr = cudaFree(devPtr);
        printf("cudaFree(NULL) returned error code: %d (%s)\n",
               freeErr, cudaGetErrorString(freeErr));
    }

    return 0;
}
```