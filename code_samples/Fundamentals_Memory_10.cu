```cuda
/*
Allocate a 2D array on the host (e.g., `M` rows, `N` cols) and represent it as a 1D array in row-major order. Allocate a corresponding 2D array on the device using `cudaMallocPitch`. Copy the host array to the device using `cudaMemcpy2D`. Explain in comments the purpose of the 'pitch' value.

Thinking:
- The program demonstrates how to map a 2D array stored in row-major order on the host to a pitched allocation on the device.
- `cudaMallocPitch` allocates memory such that each row starts at a memory address aligned to a value that maximises memory throughput (often a multiple of 256 bytes). The returned pitch is the width in bytes between successive rows.
- `cudaMemcpy2D` uses the pitch value to correctly copy data when the source and destination have different row strides. The source row stride is simply `N*sizeof(T)` because the host array is contiguous. The destination row stride is `pitch` because the device may have padding bytes at the end of each row.
- The pitch is important for performance: coalesced memory accesses require properly aligned row starts. It can also prevent misaligned memory accesses that would otherwise cause performance penalties or faults.
- This example will allocate a 4x5 float array, fill it with sample data, copy it to the device, copy it back, and print a few elements to verify correctness.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main() {
    const int M = 4; // number of rows
    const int N = 5; // number of columns
    const size_t hostRowBytes = N * sizeof(float);

    // Allocate and initialize host array as 1D in row-major order
    float *h_array = (float*)malloc(M * hostRowBytes);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_array[i * N + j] = static_cast<float>(i * N + j);
        }
    }

    // Device pitched allocation
    float *d_array;
    size_t pitch; // pitch in bytes
    CUDA_CHECK(cudaMallocPitch(&d_array, &pitch, hostRowBytes, M));

    // Copy host 2D array to device 2D array
    // Parameters:
    // - dst: pointer to device memory
    // - dstPitch: pitch of destination (bytes between rows)
    // - src: pointer to host memory
    // - srcPitch: pitch of source (bytes between rows) = hostRowBytes
    // - width: width of each row to copy (bytes)
    // - height: number of rows
    // - kind: direction of copy
    CUDA_CHECK(cudaMemcpy2D(d_array, pitch,
                            h_array, hostRowBytes,
                            hostRowBytes, M,
                            cudaMemcpyHostToDevice));

    // Optional: copy back to host to verify
    float *h_array_copy = (float*)malloc(M * hostRowBytes);
    CUDA_CHECK(cudaMemcpy2D(h_array_copy, hostRowBytes,
                            d_array, pitch,
                            hostRowBytes, M,
                            cudaMemcpyDeviceToHost));

    // Print original and copied arrays
    printf("Original host array:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%5.1f ", h_array[i * N + j]);
        }
        printf("\n");
    }

    printf("\nCopied back array from device:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%5.1f ", h_array_copy[i * N + j]);
        }
        printf("\n");
    }

    // Clean up
    free(h_array);
    free(h_array_copy);
    CUDA_CHECK(cudaFree(d_array));

    return 0;
}
```