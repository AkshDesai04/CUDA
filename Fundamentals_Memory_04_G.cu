```c
/* 
Aim of the program: 
Demonstrate that `cudaMemset` works on a 2D allocation from `cudaMallocPitch`.

Thinking behind the implementation:
1. We allocate a 2D array using `cudaMallocPitch`, which returns a device pointer
   and a pitch (row stride in bytes). The pitch is usually larger than the row
   width to satisfy alignment requirements.
2. `cudaMemset` is a linear memory operation that accepts a byte count.  By
   passing `pitch * height` we instruct it to set every byte in the entire
   pitched block, including the padding bytes inserted by the pitch.
3. After setting the memory to a known pattern (0xAA), we copy the data back to
   the host using `cudaMemcpy2D`, which respects the pitch on both source
   (device) and destination (host) sides.
4. Finally we verify that every element in the 2D array contains the expected
   pattern, proving that `cudaMemset` correctly handled the pitched allocation.
5. The program includes thorough error checking and prints the pitch,
   the contents of the host array, and a success/failure message.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA runtime API call results */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",     \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),          \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void)
{
    /* Parameters for the 2D array */
    const size_t width  = 5;   /* number of columns */
    const size_t height = 3;   /* number of rows */
    const size_t elemSize = sizeof(unsigned char); /* element type size */

    /* Device pointer and pitch (row stride in bytes) */
    unsigned char *devPtr = NULL;
    size_t pitch = 0;

    /* Allocate pitched 2D memory on the device */
    CHECK_CUDA(cudaMallocPitch((void **)&devPtr, &pitch, width * elemSize, height));
    printf("Allocated pitched 2D memory: width=%zu, height=%zu, elemSize=%zu bytes\n",
           width, height, elemSize);
    printf("Pitch (bytes) = %zu\n", pitch);

    /* Set all bytes in the pitched block to 0xAA using cudaMemset */
    const unsigned char value = 0xAA;
    size_t bytesToSet = pitch * height;
    CHECK_CUDA(cudaMemset(devPtr, value, bytesToSet));
    printf("Called cudaMemset to set %zu bytes to 0x%02X\n", bytesToSet, value);

    /* Allocate host memory to receive the data */
    unsigned char *hostArray = (unsigned char *)malloc(width * height * elemSize);
    if (!hostArray) {
        fprintf(stderr, "Failed to allocate host memory\n");
        cudaFree(devPtr);
        return EXIT_FAILURE;
    }

    /* Copy the pitched device memory back to host 2D array */
    CHECK_CUDA(cudaMemcpy2D(hostArray, width * elemSize,
                            devPtr, pitch,
                            width * elemSize, height,
                            cudaMemcpyDeviceToHost));
    printf("Copied data back to host.\n");

    /* Print the host array and verify contents */
    printf("Host array contents:\n");
    int allMatch = 1;
    for (size_t row = 0; row < height; ++row) {
        for (size_t col = 0; col < width; ++col) {
            unsigned char hostVal = hostArray[row * width + col];
            printf("0x%02X ", hostVal);
            if (hostVal != value) allMatch = 0;
        }
        printf("\n");
    }

    if (allMatch) {
        printf("SUCCESS: All elements match the expected value 0x%02X.\n", value);
    } else {
        printf("FAILURE: Some elements did not match the expected value 0x%02X.\n", value);
    }

    /* Clean up */
    free(hostArray);
    CHECK_CUDA(cudaFree(devPtr));

    return EXIT_SUCCESS;
}
```