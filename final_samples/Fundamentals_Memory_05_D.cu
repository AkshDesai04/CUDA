```cpp
/*
Aim of the program: Try to call `cudaFree` on a host pointer from `malloc`. What error does this produce?

Thinking process:
- The CUDA runtime expects `cudaFree` to be called on memory that was allocated with `cudaMalloc` or
  `cudaHostAlloc` (or other CUDA allocation routines). A pointer obtained from standard C `malloc`
  is a host memory pointer that has no association with the GPU.
- According to the CUDA Runtime API documentation, passing a non-device pointer to `cudaFree`
  should result in the error code `cudaErrorInvalidDevicePointer` (error code 17).
- To verify this, we write a small CUDA C program that:
  1. Allocates a block of host memory using `malloc`.
  2. Attempts to free that block with `cudaFree`.
  3. Checks the returned error code and prints it using `cudaGetErrorString`.
  4. Cleans up the host memory with `free` afterwards.
- When compiled and executed, the program should print an error message similar to:
  "cudaFree error: invalid device pointer".
- The program demonstrates that `cudaFree` cannot be used to free host memory allocated
  with `malloc`, and shows the specific CUDA error returned.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(void) {
    const size_t size = 1024;
    // Allocate host memory with malloc
    void *host_ptr = malloc(size);
    if (!host_ptr) {
        fprintf(stderr, "Failed to allocate host memory with malloc.\n");
        return EXIT_FAILURE;
    }

    // Attempt to free the host memory using cudaFree
    cudaError_t err = cudaFree(host_ptr);

    // Print the error returned by cudaFree
    if (err != cudaSuccess) {
        printf("cudaFree error: %s (code %d)\n", cudaGetErrorString(err), err);
    } else {
        printf("cudaFree succeeded unexpectedly.\n");
    }

    // Clean up the host memory
    free(host_ptr);

    return EXIT_SUCCESS;
}
```