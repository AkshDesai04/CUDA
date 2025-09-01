```cuda
/* 
Aim: Try to call `cudaFree` on a host pointer from `malloc`. What error does this produce?

Thinking:
1. Allocate memory on the host using `malloc`. This memory is ordinary CPU memory, not device memory.
2. Attempt to free that host pointer with `cudaFree`, which is designed to free device memory allocated by `cudaMalloc` or similar functions.
3. According to the CUDA Runtime API documentation, passing a non-device pointer to `cudaFree` should return an error code indicating an invalid device pointer. In the current CUDA runtime, this is `cudaErrorInvalidDevicePointer` (error code 3).
4. The program will capture the error returned by `cudaFree`, print the error string using `cudaGetErrorString`, and exit.
5. This demonstrates that `cudaFree` cannot be used to free host memory, and the runtime reports the appropriate error.

The code below implements this test and prints the error information. */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(void) {
    /* Allocate host memory */
    float *host_ptr = (float *)malloc(100 * sizeof(float));
    if (!host_ptr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Attempt to free host memory with cudaFree */
    cudaError_t err = cudaFree(host_ptr);

    /* Print the error */
    printf("Attempted cudaFree on host pointer returned error code: %d\n", err);
    printf("Error string: %s\n", cudaGetErrorString(err));

    /* Clean up host memory */
    free(host_ptr);

    return (err == cudaSuccess) ? EXIT_SUCCESS : EXIT_FAILURE;
}
```