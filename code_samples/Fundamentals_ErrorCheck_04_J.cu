```cpp
/*
Pass a size of 0 to `cudaMemcpy`. Is this an error?

Thinking:
The CUDA Runtime API specifies that the function cudaMemcpy copies a region of memory from a source to a destination.
The function signature is:
    cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);

According to the documentation, passing a count of 0 should be allowed and is defined as a no-op.
It does not perform any memory copy, but it should still return cudaSuccess as long as the pointers are
valid (or can be NULL because no dereference occurs). In practice, most samples show that a zero-size
transfer is treated as a successful operation and is not considered an error.

To verify this, the program:
1. Allocates a small block of host memory and a device memory block.
2. Calls cudaMemcpy with count=0.
3. Checks the returned cudaError_t.
4. Prints whether the operation succeeded or failed.

We also handle errors for the memory allocation calls, ensuring that if something fails there
we report it. Finally we free the allocated memory and exit.

The expected output is that the zero-size cudaMemcpy call returns cudaSuccess, confirming
that it is not an error.

This program can be compiled with:
    nvcc -o test_cudamemcpy test_cudamemcpy.cu
and run to observe the behavior.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Utility macro for error checking
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                           \
    } while (0)

int main(void) {
    // Allocate host memory
    int *h_data = (int*)malloc(4 * sizeof(int));
    if (!h_data) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_data = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_data, 4 * sizeof(int)));

    // Perform cudaMemcpy with size 0
    cudaError_t mem_err = cudaMemcpy(d_data, h_data, 0, cudaMemcpyHostToDevice);

    // Check the result
    if (mem_err == cudaSuccess) {
        printf("cudaMemcpy with size 0 succeeded (no operation performed).\n");
    } else {
        printf("cudaMemcpy with size 0 returned error: %s\n", cudaGetErrorString(mem_err));
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);

    return EXIT_SUCCESS;
}
```