/*
Aim: Trigger `cudaErrorInvalidValue` by passing an invalid `kind` to `cudaMemcpy`, e.g., `(cudaMemcpyKind)123`.

The goal of this program is to illustrate how CUDA handles an incorrect `cudaMemcpyKind` value. By casting an arbitrary integer (123) to the `cudaMemcpyKind` enumeration, the runtime will detect that the value is not one of the valid kinds (e.g., `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice`, etc.). The CUDA runtime is expected to return the error code `cudaErrorInvalidValue`. We capture this error, print a descriptive message, and clean up allocated resources.

Key steps:
1. Allocate a small buffer on the host and a corresponding buffer on the device.
2. Attempt a memory copy using an invalid `cudaMemcpyKind` value.
3. Check the return code from `cudaMemcpy`. If it indicates failure, we print the error string.
4. Release any allocated memory before exiting.

This demonstrates error handling and is a minimal, self‑contained example that compiles with `nvcc` and runs on a system with CUDA support.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    const size_t N = 10;
    const size_t bytes = N * sizeof(int);

    int *h_data = (int *)malloc(bytes);
    int *d_data = NULL;

    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }

    // Initialize host data (optional)
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = (int)i;
    }

    // Allocate device memory
    cudaError_t err = cudaMalloc((void **)&d_data, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_data);
        return 1;
    }

    // Intentionally use an invalid cudaMemcpyKind value
    err = cudaMemcpy(d_data, h_data, bytes, (cudaMemcpyKind)123);

    if (err != cudaSuccess) {
        printf("cudaMemcpy returned error: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaMemcpy succeeded (unexpected).\n");
    }

    // Clean up
    cudaFree(d_data);
    free(h_data);

    return 0;
}
