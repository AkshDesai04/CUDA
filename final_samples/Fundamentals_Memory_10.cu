```cuda
/*
Allocate a 2D array on the host (e.g., M rows, N cols) and represent it as a 1D array in row-major order.
Allocate a corresponding 2D array on the device using cudaMallocPitch.
Copy the host array to the device using cudaMemcpy2D.
Explain in comments the purpose of the 'pitch' value.

Thinking process:
1. Define constants M and N for number of rows and columns.
2. Allocate a single contiguous 1D array on the host with M*N elements.
3. Fill the host array with sample data (e.g., row-major index values).
4. Allocate device memory for a 2D array using cudaMallocPitch, which returns a pointer to device memory
   and the pitch (bytes per row, possibly padded to satisfy alignment requirements).
5. Use cudaMemcpy2D to copy the 2D data from host to device, providing source and destination pitches.
6. Include comments explaining that 'pitch' is the actual allocated width of each row on the device
   (including any padding) and is necessary for correct indexing and memory layout.
7. (Optional) Copy back to host and verify contents.
8. Free host and device memory and handle any CUDA errors.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define M  4   // number of rows
#define N  5   // number of columns

// Simple error checking macro
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int main(void)
{
    size_t hostSize = M * N * sizeof(float);
    float *h_array = (float*)malloc(hostSize);
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array in row-major order
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            h_array[row * N + col] = (float)(row * N + col);
        }
    }

    // Device 2D array allocation with pitch
    float *d_array;
    size_t pitch; // pitch is the width in bytes of each row on the device
    CHECK_CUDA(cudaMallocPitch((void**)&d_array, &pitch, N * sizeof(float), M));

    // Explanation of pitch:
    // CUDA may pad each row in device memory to align to 256-byte boundaries (or other alignment)
    // for optimal memory throughput. 'pitch' represents the actual number of bytes between the start
    // of consecutive rows in device memory, which may be larger than N * sizeof(float).
    // When copying or accessing 2D data, we must use this pitch value to correctly calculate offsets.

    // Copy from host to device using cudaMemcpy2D
    // srcPitch is the number of bytes between successive rows in host array (contiguous)
    // dstPitch is the pitch returned by cudaMallocPitch
    CHECK_CUDA(cudaMemcpy2D(d_array, pitch,
                            h_array, N * sizeof(float),
                            N * sizeof(float), M,
                            cudaMemcpyHostToDevice));

    // (Optional) Verify by copying back to host
    float *h_back = (float*)malloc(hostSize);
    if (!h_back) {
        fprintf(stderr, "Failed to allocate host memory for copy back\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaMemcpy2D(h_back, N * sizeof(float),
                            d_array, pitch,
                            N * sizeof(float), M,
                            cudaMemcpyDeviceToHost));

    // Print original and copied data to verify correctness
    printf("Original host array:\n");
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            printf("%6.1f ", h_array[row * N + col]);
        }
        printf("\n");
    }

    printf("\nCopied back array:\n");
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            printf("%6.1f ", h_back[row * N + col]);
        }
        printf("\n");
    }

    // Clean up
    free(h_array);
    free(h_back);
    CHECK_CUDA(cudaFree(d_array));

    return 0;
}
```