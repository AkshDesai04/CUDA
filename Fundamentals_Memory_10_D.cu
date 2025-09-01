/*
Perform a 2D device-to-device copy using `cudaMemcpy2D`.

Thinking Process:
1. Choose a simple 2D array example (e.g., 16 columns × 8 rows of integers).
2. Allocate pitched memory on the device for both source and destination using `cudaMallocPitch`.
3. Initialize the source array on the host, copy it to the device source array.
4. Use `cudaMemcpy2D` with `cudaMemcpyDeviceToDevice` to copy from the source device pointer to the destination device pointer, preserving the pitch.
5. Copy the destination device array back to the host to verify the copy.
6. Print a few elements to confirm the operation succeeded.
7. Include error checking for all CUDA calls.
8. Keep the code self‑contained, compile-able with `nvcc`, and free all allocated memory before exiting.
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void)
{
    // Define 2D dimensions
    const size_t width  = 16;   // number of columns
    const size_t height = 8;    // number of rows
    const size_t elemSize = sizeof(int);

    // Host allocation
    int *h_src = (int*)malloc(width * height * elemSize);
    int *h_dst = (int*)malloc(width * height * elemSize);

    // Initialize source data on host
    for (size_t row = 0; row < height; ++row)
        for (size_t col = 0; col < width; ++col)
            h_src[row * width + col] = (int)(row * width + col);

    // Device pitched allocation
    int *d_src = NULL;
    int *d_dst = NULL;
    size_t srcPitch = 0;
    size_t dstPitch = 0;

    CHECK_CUDA(cudaMallocPitch((void**)&d_src, &srcPitch, width * elemSize, height));
    CHECK_CUDA(cudaMallocPitch((void**)&d_dst, &dstPitch, width * elemSize, height));

    // Copy source data from host to device
    CHECK_CUDA(cudaMemcpy2D(d_src, srcPitch,
                            h_src, width * elemSize,
                            width * elemSize, height,
                            cudaMemcpyHostToDevice));

    // Perform device-to-device 2D copy
    CHECK_CUDA(cudaMemcpy2D(d_dst, dstPitch,
                            d_src, srcPitch,
                            width * elemSize, height,
                            cudaMemcpyDeviceToDevice));

    // Copy result back to host for verification
    CHECK_CUDA(cudaMemcpy2D(h_dst, width * elemSize,
                            d_dst, dstPitch,
                            width * elemSize, height,
                            cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("Sample verification (row, col) : value\n");
    for (size_t row = 0; row < height; ++row) {
        for (size_t col = 0; col < width; ++col) {
            if (h_dst[row * width + col] != h_src[row * width + col]) {
                printf("Mismatch at (%zu,%zu)! src=%d dst=%d\n",
                       row, col, h_src[row * width + col], h_dst[row * width + col]);
                return EXIT_FAILURE;
            }
        }
    }
    printf("All values match. 2D device-to-device copy succeeded.\n");

    // Clean up
    free(h_src);
    free(h_dst);
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    return EXIT_SUCCESS;
}
