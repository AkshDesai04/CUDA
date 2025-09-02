```cpp
/*
Use `cudaMemset2D` to initialize a pitched 2D allocation to zero.

Thinking:
The task is to demonstrate the use of cudaMemset2D, which sets a 2‑D memory region
to a specified value, correctly handling pitched memory allocations on the device.
To achieve this we will:

1. Define the dimensions of a 2‑D array (e.g., width = 256, height = 128) and
   the element type (float).
2. Allocate pitched memory on the device using cudaMallocPitch. This returns a
   device pointer and the pitch (bytes per row).
3. Use cudaMemset2D to set the entire 2‑D region to zero. The width parameter
   for cudaMemset2D is specified in bytes, so we multiply the element width
   by the array width.
4. Allocate a corresponding host array and copy the device memory back to the
   host with cudaMemcpy2D to verify that the data has been zeroed.
5. Print a few elements from the host array to confirm they are zero.
6. Perform proper error checking after each CUDA API call, and clean up
   allocated resources before exiting.

The program is written in CUDA C (file extension .cu) and can be compiled with
`nvcc`. It demonstrates the basic pattern for handling pitched allocations
and resetting them with cudaMemset2D.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                                \
    } while (0)

int main(void)
{
    const size_t width  = 256;   // number of elements per row
    const size_t height = 128;   // number of rows
    const size_t elemSize = sizeof(float);  // element size

    // Device pointer and pitch
    float *d_ptr = nullptr;
    size_t pitch = 0;

    // Allocate pitched memory on the device
    CHECK_CUDA(cudaMallocPitch((void**)&d_ptr, &pitch, width * elemSize, height));

    // Initialize the pitched 2D allocation to zero using cudaMemset2D
    // width in bytes, pitch in bytes, value is 0
    CHECK_CUDA(cudaMemset2D(d_ptr, pitch, 0, width * elemSize, height));

    // Allocate host memory to copy back the data
    float *h_ptr = (float*)malloc(height * pitch); // allocate with device pitch for simplicity
    if (!h_ptr) {
        fprintf(stderr, "Host allocation failed\n");
        cudaFree(d_ptr);
        return EXIT_FAILURE;
    }

    // Copy the device data back to host
    CHECK_CUDA(cudaMemcpy2D(h_ptr, pitch,
                            d_ptr, pitch,
                            width * elemSize, height,
                            cudaMemcpyDeviceToHost));

    // Verify by printing the first few elements
    bool allZero = true;
    for (size_t i = 0; i < height && allZero; ++i) {
        for (size_t j = 0; j < width; ++j) {
            float val = h_ptr[i * pitch / elemSize + j];
            if (val != 0.0f) {
                allZero = false;
                printf("Non-zero value found at (%zu, %zu): %f\n", i, j, val);
                break;
            }
        }
    }
    if (allZero) {
        printf("All elements are zero as expected.\n");
    }

    // Clean up
    free(h_ptr);
    CHECK_CUDA(cudaFree(d_ptr));

    return 0;
}
```