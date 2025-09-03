/*
Aim: Try to call `cudaFree` on a host pointer from `malloc`. What error does this produce?

Thought process:
- The objective is to allocate memory on the host using standard C `malloc`, then attempt to free that memory using CUDA's `cudaFree` API, which is designed for freeing device memory.
- `cudaFree` performs checks to ensure that the pointer being freed was allocated by a CUDA runtime function such as `cudaMalloc`. When a host pointer is passed, the CUDA runtime does not recognize it as a valid device pointer.
- This mismatch leads to an error code being returned by `cudaFree`. The expected error is `cudaErrorInvalidDevicePointer`, which corresponds to the error string "Invalid device pointer".
- To demonstrate this, we will:
  1. Allocate a small block of memory on the host with `malloc`.
  2. Call `cudaFree` on that host pointer.
  3. Capture the error code returned by `cudaFree`.
  4. Use `cudaGetErrorString` to convert the error code to a human‑readable message and print it.
- This simple program will compile and run on any system with CUDA installed, and will clearly show the error that occurs when attempting to free a host pointer with `cudaFree`.

The code below implements this plan in CUDA C.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

int main(void) {
    // Allocate host memory using malloc
    size_t size = 256 * sizeof(float);
    float *hostPtr = (float *)malloc(size);
    if (hostPtr == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Attempt to free the host pointer using cudaFree
    cudaError_t err = cudaFree((void *)hostPtr);

    // Print the error code and message
    if (err != cudaSuccess) {
        printf("cudaFree returned error code %d: %s\n",
               err, cudaGetErrorString(err));
    } else {
        printf("cudaFree succeeded (unexpected).\n");
    }

    // Clean up host memory
    free(hostPtr);

    return 0;
}
