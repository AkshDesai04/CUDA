```cuda
/*
 * Is `cudaDeviceReset()` necessary for a program that cleans up all its resources properly with `cudaFree`? Explain in a comment.
 *
 * My thinking:
 * - `cudaDeviceReset()` performs a device-wide reset, shutting down the device, flushing any pending work,
 *   destroying all resources associated with the current context, and restoring the device to a clean state.
 * - If a program frees all device memory with `cudaFree` (and also frees any streams, events, etc.) then, for that
 *   particular execution, it has effectively released all GPU resources. In this case, when the process exits,
 *   the CUDA runtime automatically destroys the context and frees resources, so an explicit `cudaDeviceReset()`
 *   is not required for correctness or to avoid memory leaks.
 * - However, `cudaDeviceReset()` is useful when you want to ensure the device is in a clean state before
 *   launching another unrelated CUDA program in the same process or after an error that leaves the context
 *   in an inconsistent state. It also helps in debugging and in cases where the application continues
 *   running and might allocate new contexts later.
 * - In summary, for a short-lived program that properly frees all resources and terminates, `cudaDeviceReset()`
 *   is not strictly necessary. For long-lived processes, error recovery, or when you need a guaranteed clean
 *   state for subsequent work, you should call it.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    const int N = 10;
    int *d_arr;
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc((void**)&d_arr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // (Optional) Do some work on the GPU here...

    // Free device memory
    err = cudaFree(d_arr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // No explicit cudaDeviceReset() needed for this short program.
    // The runtime will clean up the context on process exit.

    printf("Device memory allocated and freed successfully.\n");
    return 0;
}
```