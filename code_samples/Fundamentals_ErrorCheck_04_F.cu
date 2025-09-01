/*
 * Look up the documentation for `cudaErrorInvalidValue` and find another API call that can trigger it.
 *
 * In order to satisfy the developer request for a complete CUDA C program, I will embed this statement
 * verbatim at the beginning of the file as required.  The developer also asked that I include all
 * of my own thinking in the same comment block.  To accomplish that, I describe the reasoning
 * behind choosing an API that will trigger the `cudaErrorInvalidValue` error code.
 *
 * The CUDA Runtime API documents the error code `cudaErrorInvalidValue` as being returned by
 * several functions when an argument passed to them is not valid.  One common scenario that
 * produces this error is passing a `NULL` device pointer to `cudaMemcpy` (or any of its
 * variants).  The documentation for `cudaMemcpy` explicitly states that if either `src` or
 * `dst` is `NULL` or not a valid device pointer, the function returns `cudaErrorInvalidValue`.
 *
 * Thus, I will create a simple program that deliberately passes a `NULL` pointer to
 * `cudaMemcpy`.  The program will then print out the returned error string so that we can
 * see the `cudaErrorInvalidValue` message.  The code is minimal, uses only the CUDA
 * runtime API, and is written in C syntax for compatibility with .cu files.
 *
 * The program follows these steps:
 * 1. Allocate a small host buffer (though it will not be used for the copy).
 * 2. Call `cudaMemcpy` with a `NULL` destination pointer and a non-zero byte count.
 * 3. Capture the returned `cudaError_t` and print its description using `cudaGetErrorString`.
 * 4. Exit with a status indicating success or failure.
 *
 * This demonstrates an alternative API call (`cudaMemcpy`) that triggers the same
 * error code as other invalid-parameter situations described in the documentation.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    // Allocate a small host buffer (not used in the bad copy)
    size_t hostSize = 10;
    char *hostBuf = (char *)malloc(hostSize);
    if (hostBuf == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }

    // Intentionally pass a NULL destination pointer to cudaMemcpy to trigger cudaErrorInvalidValue
    cudaError_t err = cudaMemcpy(NULL, hostBuf, hostSize, cudaMemcpyHostToDevice);

    // Print the error message
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy returned error: %s\n", cudaGetErrorString(err));
        // Clean up and exit with error
        free(hostBuf);
        return 1;
    } else {
        printf("cudaMemcpy succeeded unexpectedly.\n");
    }

    // Clean up
    free(hostBuf);
    return 0;
}
