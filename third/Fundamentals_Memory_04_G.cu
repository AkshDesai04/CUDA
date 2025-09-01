```cpp
/*
Aim: Demonstrate that `cudaMemset` works on a 2D allocation from `cudaMallocPitch`.

Thinking:
The goal is to allocate a 2D array on the device using `cudaMallocPitch`, which
returns a device pointer and a pitch (the number of bytes per row, including any
padding). `cudaMemset` is a byte-wise set operation: it writes a single byte
value to a contiguous block of memory. When dealing with pitched memory, the
total size of the memory region that needs to be initialized is the pitch
multiplied by the number of rows. Simply passing width*height to `cudaMemset`
would leave the padding bytes uninitialized, which can be misleading. Therefore
the correct way is to call `cudaMemset(dptr, value, pitch * height)`.

To verify this, we will:
1. Allocate a 2D array of unsigned chars (size 5x10) using `cudaMallocPitch`.
2. Use `cudaMemset` to set every byte in the allocated region to 0xAB.
3. Copy the pitched memory back to the host using `cudaMemcpy2D`.
4. Print the host array to show that all elements (including those that would
   reside in padding areas if they existed) contain 0xAB.

The program includes basic error checking and prints the resulting array.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main(void) {
    const size_t width = 5;   // number of columns
    const size_t height = 10; // number of rows
    const size_t elemSize = sizeof(unsigned char);

    unsigned char *d_ptr = nullptr;
    size_t pitch = 0; // pitch in bytes

    // Allocate pitched memory on the device
    CUDA_CHECK(cudaMallocPitch(&d_ptr, &pitch, width * elemSize, height));

    printf("Allocated pitched memory: pitch = %zu bytes\n", pitch);

    // Set the entire allocated region to 0xAB
    // Count = pitch * height (bytes)
    CUDA_CHECK(cudaMemset(d_ptr, 0xAB, pitch * height));

    // Allocate host memory
    unsigned char *h_arr = (unsigned char*)malloc(width * height * elemSize);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Copy the pitched device memory back to host
    CUDA_CHECK(cudaMemcpy2D(h_arr,
                            width * elemSize,            // host pitch
                            d_ptr,
                            pitch,                      // device pitch
                            width * elemSize,            // width in bytes
                            height,
                            cudaMemcpyDeviceToHost));

    // Print the host array to verify
    printf("Host array after cudaMemcpy2D (should all be 0xAB):\n");
    for (size_t row = 0; row < height; ++row) {
        for (size_t col = 0; col < width; ++col) {
            printf("%02X ", h_arr[row * width + col]);
        }
        printf("\n");
    }

    // Clean up
    free(h_arr);
    CUDA_CHECK(cudaFree(d_ptr));

    printf("Demonstration completed successfully.\n");
    return EXIT_SUCCESS;
}
```