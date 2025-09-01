/*
Use `cudaMemGetInfo` to find out how much free memory is available, then try to allocate slightly more than that amount to trigger the error.

Thinking:
- First, query free and total memory via cudaMemGetInfo.
- Print both values.
- Compute a size slightly larger than free: free + 1MB (or a fraction).
- Try cudaMalloc with that size. This should fail because request > free memory.
- Capture error returned by cudaMalloc.
- Print error code and description.
- Optionally, also attempt to allocate an enormous amount to confirm failure.
- Ensure cleanup: if any allocation succeeded, free it.
- Use cudaDeviceReset before exit.
- Handle possible error on cudaMemGetInfo itself.
- Use 64-bit integers (size_t) for sizes.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Free memory:  %zu bytes (%.2f MB)\n", free_mem, free_mem / (1024.0 * 1024.0));
    printf("Total memory: %zu bytes (%.2f MB)\n", total_mem, total_mem / (1024.0 * 1024.0));

    // Allocate slightly more than free memory to trigger an error
    size_t request_size = free_mem + (size_t)(1 * 1024 * 1024); // +1 MB
    printf("\nAttempting to allocate %zu bytes (%.2f MB) ...\n", request_size, request_size / (1024.0 * 1024.0));

    void *d_ptr = NULL;
    err = cudaMalloc(&d_ptr, request_size);
    if (err != cudaSuccess)
    {
        printf("cudaMalloc failed as expected.\n");
        printf("Error code: %d (%s)\n", err, cudaGetErrorString(err));
    }
    else
    {
        printf("Unexpectedly succeeded in allocation.\n");
        // If allocation succeeded, free the memory
        cudaFree(d_ptr);
    }

    // Optional: attempt an obviously huge allocation to further confirm failure
    size_t huge_request = (size_t)4 * total_mem; // 4x total memory
    printf("\nAttempting to allocate an obviously huge size %zu bytes ...\n", huge_request);
    void *d_ptr2 = NULL;
    err = cudaMalloc(&d_ptr2, huge_request);
    if (err != cudaSuccess)
    {
        printf("Huge allocation failed as expected.\n");
        printf("Error code: %d (%s)\n", err, cudaGetErrorString(err));
    }
    else
    {
        printf("Unexpectedly succeeded in huge allocation.\n");
        cudaFree(d_ptr2);
    }

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
