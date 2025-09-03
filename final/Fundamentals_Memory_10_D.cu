/*
Perform a 2D device-to-device copy using `cudaMemcpy2D`.

This program demonstrates how to allocate two-dimensional arrays on the GPU using
`cudaMallocPitch`, initialize one of them with a simple pattern via a CUDA kernel,
and then copy the contents from the source array to a destination array using
`cudaMemcpy2D`. After the copy, the destination data is transferred back to
the host and printed to verify that the operation succeeded.

Key steps:
1. Define image dimensions and pitch for device memory allocation.
2. Allocate source and destination buffers with `cudaMallocPitch`.
3. Launch a kernel to initialize the source buffer with a unique value per
   pixel so we can easily distinguish it later.
4. Use `cudaMemcpy2D` to copy from the source to the destination buffer.
   The source and destination pitches, width in bytes, and height are all
   specified explicitly.
5. Copy the destination buffer back to the host with `cudaMemcpy2D`.
6. Print a few pixel values to confirm the copy.

The code includes error checking via a simple macro `CHECK_CUDA` to ensure
that each CUDA API call succeeds, simplifying debugging and providing
informative error messages. The program can be compiled with `nvcc`:

    nvcc -o copy2d copy2d.cu

and run on any machine with an appropriate CUDA-capable GPU.

*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

/* Kernel to initialize a 2D array with a value that depends on its coordinates */
__global__ void initKernel(unsigned char *data, size_t pitch, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (x < width && y < height) {
        unsigned char *row = (unsigned char *)((char *)data + y * pitch);
        row[x] = (unsigned char)(x + y * width); // simple pattern
    }
}

int main(void)
{
    /* Image dimensions */
    const int width  = 8;   // width in pixels
    const int height = 6;   // height in pixels

    /* Size of each element (bytes) */
    const size_t elemSize = sizeof(unsigned char);

    /* Allocate source and destination arrays on device using pitched allocation */
    unsigned char *d_src = NULL;
    unsigned char *d_dst = NULL;
    size_t pitch_src = 0;
    size_t pitch_dst = 0;

    CHECK_CUDA(cudaMallocPitch(&d_src, &pitch_src, width * elemSize, height));
    CHECK_CUDA(cudaMallocPitch(&d_dst, &pitch_dst, width * elemSize, height));

    /* Initialize source array with a pattern */
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    initKernel<<<gridDim, blockDim>>>(d_src, pitch_src, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Perform device-to-device 2D copy using cudaMemcpy2D */
    CHECK_CUDA(cudaMemcpy2D(d_dst,
                            pitch_dst,
                            d_src,
                            pitch_src,
                            width * elemSize,
                            height,
                            cudaMemcpyDeviceToDevice));

    /* Copy destination array back to host for verification */
    unsigned char *h_dst = (unsigned char *)malloc(width * height * elemSize);
    CHECK_CUDA(cudaMemcpy2D(h_dst,
                            width * elemSize,
                            d_dst,
                            pitch_dst,
                            width * elemSize,
                            height,
                            cudaMemcpyDeviceToHost));

    /* Print a few pixels to verify copy */
    printf("First few pixels of destination array:\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%3d ", h_dst[y * width + x]);
        }
        printf("\n");
    }

    /* Clean up */
    free(h_dst);
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
