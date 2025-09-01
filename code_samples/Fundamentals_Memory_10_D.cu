```cuda
/*
Perform a 2D device-to-device copy using `cudaMemcpy2D`.

Thinking:
1. The program will allocate two 2D arrays on the device using cudaMallocPitch so that the pitch is handled automatically.
2. We will fill a host array with known data, copy it to the source device array.
3. Then we will perform a device-to-device copy from the source array to the destination array using cudaMemcpy2D.
4. Finally, we copy the destination back to host to verify the copy and print a few elements.
5. Error checking is done via a helper macro to keep the code clean.
6. The program is self‑contained and can be compiled with `nvcc -o copy2d copy2d.cu`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                           \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",           \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                   \
        }                                                                          \
    } while (0)

int main(void) {
    const size_t width = 16;   // number of columns
    const size_t height = 8;   // number of rows
    const size_t elemSize = sizeof(int);

    // Host arrays
    int *h_src = (int *)malloc(width * height * elemSize);
    int *h_dst = (int *)malloc(width * height * elemSize);
    if (!h_src || !h_dst) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize host source with a simple pattern
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            h_src[y * width + x] = (int)(y * width + x);
        }
    }

    // Device pointers and pitches
    int *d_src = NULL;
    int *d_dst = NULL;
    size_t srcPitch = 0, dstPitch = 0;

    // Allocate pitched memory for source
    CUDA_CHECK(cudaMallocPitch(&d_src, &srcPitch, width * elemSize, height));
    // Allocate pitched memory for destination
    CUDA_CHECK(cudaMallocPitch(&d_dst, &dstPitch, width * elemSize, height));

    // Copy host source to device source
    CUDA_CHECK(cudaMemcpy2D(d_src, srcPitch,
                            h_src, width * elemSize,
                            width * elemSize, height,
                            cudaMemcpyHostToDevice));

    // Perform device-to-device copy
    CUDA_CHECK(cudaMemcpy2D(d_dst, dstPitch,
                            d_src, srcPitch,
                            width * elemSize, height,
                            cudaMemcpyDeviceToDevice));

    // Copy back from device destination to host
    CUDA_CHECK(cudaMemcpy2D(h_dst, width * elemSize,
                            d_dst, dstPitch,
                            width * elemSize, height,
                            cudaMemcpyDeviceToHost));

    // Verify some elements
    printf("Verifying copied data:\n");
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            int val_src = h_src[y * width + x];
            int val_dst = h_dst[y * width + x];
            if (val_src != val_dst) {
                printf("Mismatch at (%zu,%zu): src=%d dst=%d\n", y, x, val_src, val_dst);
            }
        }
    }
    printf("Verification complete.\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    free(h_src);
    free(h_dst);

    return EXIT_SUCCESS;
}
```