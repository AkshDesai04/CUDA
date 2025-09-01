```cpp
/* 
Try to call `cudaFree` on a host pointer from `malloc`. What error does this produce?

Thinking:
- The aim is to demonstrate the behavior of calling cudaFree on a host pointer.
- Allocate memory on the host using standard C `malloc`.
- Pass that host pointer to `cudaFree` which expects a device pointer.
- Capture the returned cudaError_t value.
- Use `cudaGetErrorString` to obtain a human-readable description.
- Print both the numeric error code and the description.
- According to CUDA documentation, passing an invalid (host) pointer to cudaFree
  should yield `CUDA_ERROR_INVALID_DEVICE_POINTER` (error code 11).
- The program will compile with nvcc and run on a system with CUDA installed.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(void) {
    // Allocate memory on the host
    size_t size = 256;
    void *hostPtr = malloc(size);
    if (hostPtr == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Attempt to free the host pointer using cudaFree
    cudaError_t err = cudaFree(hostPtr);

    // Check the result
    if (err != cudaSuccess) {
        printf("cudaFree failed with error code %d: %s\n", err, cudaGetErrorString(err));
    } else {
        printf("cudaFree succeeded (unexpected).\n");
    }

    // Clean up the host memory
    free(hostPtr);

    return 0;
}
```