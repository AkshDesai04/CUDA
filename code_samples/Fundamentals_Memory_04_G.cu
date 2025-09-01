```c
/*
Aim: Demonstrate that cudaMemset works on a 2D allocation from cudaMallocPitch.

Thinking:
1. Allocate a 2D array of ints using cudaMallocPitch. Pitch may be larger than the logical width * sizeof(int) due to alignment, so each row has a potential padding area.
2. Use cudaMemset to set all bytes in the allocated memory to a specific value (e.g., 0xAB). Since cudaMemset works on byte granularity, each int will be filled with the pattern 0xABABABAB (in little-endian).
3. Copy the pitched memory back to a linear host array to verify the contents. When copying, use cudaMemcpy2D with the same pitch.
4. Print a few elements to confirm the memset worked across rows and that the padding bytes (if any) are also set (they may not be copied back if we only copy the logical width).
5. Use a simple error-checking macro to keep the code clean.
6. Clean up by freeing device and host memory.

The program will be self-contained, compilable with nvcc, and will clearly show that cudaMemset correctly initializes a 2D pitched allocation.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

int main(void) {
    const int width = 5;      // number of elements per row
    const int height = 4;     // number of rows
    const size_t elemSize = sizeof(int);

    size_t pitch;             // number of bytes between starts of consecutive rows
    int *devPtr = NULL;

    // Allocate pitched 2D memory on device
    CHECK(cudaMallocPitch(&devPtr, &pitch, width * elemSize, height));

    // Display pitch
    printf("Allocated pitch: %zu bytes\n", pitch);

    // Use cudaMemset to set all bytes to 0xAB
    const int byteValue = 0xAB;
    size_t totalSize = pitch * height;
    CHECK(cudaMemset(devPtr, byteValue, totalSize));

    // Allocate host array to copy back
    int *hostPtr = (int*)malloc(width * height * elemSize);
    if (!hostPtr) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Copy back 2D memory from device to host
    CHECK(cudaMemcpy2D(hostPtr, width * elemSize,
                       devPtr, pitch,
                       width * elemSize, height,
                       cudaMemcpyDeviceToHost));

    // Verify contents
    printf("Verifying data after cudaMemset:\n");
    for (int row = 0; row < height; ++row) {
        printf("Row %d: ", row);
        for (int col = 0; col < width; ++col) {
            int val = hostPtr[row * width + col];
            // Since we set all bytes to 0xAB, each int should be 0xABABABAB
            printf("%08X ", (unsigned int)val);
        }
        printf("\n");
    }

    // Clean up
    free(hostPtr);
    CHECK(cudaFree(devPtr));

    printf("Program completed successfully.\n");
    return EXIT_SUCCESS;
}
```