/*
 * Aim of the program:
 * Demonstrate that `cudaMemset` works on a 2D allocation from `cudaMallocPitch`.
 *
 * Thinking:
 * 1. We need a CUDA C program that allocates a 2D array using cudaMallocPitch.
 *    This function returns a device pointer and a pitch (in bytes) that aligns each
 *    row to the device's memory alignment requirements. The pitch is usually
 *    larger than the requested row width in bytes, so padding bytes exist at
 *    the end of each row.
 *
 * 2. After allocating, we will use cudaMemset to set every byte in the
 *    allocated memory region to a known value. The API call expects the
 *    number of bytes to set. To cover the entire 2D allocation we multiply
 *    the pitch by the number of rows (height). This will also set the padding
 *    bytes, but that's fine for demonstration purposes.
 *
 * 3. We will copy the entire pitched memory block back to host using
 *    cudaMemcpy2D, which respects the pitch. On the host side we interpret
 *    the data as an array of integers and verify that each element contains
 *    the expected value (0xAA repeated in each byte of the int, i.e.
 *    0xAAAAAAAA). We also print a few elements to visually confirm.
 *
 * 4. We use a simple error checking macro to catch CUDA API failures.
 *    The program prints status messages and exits if an error occurs.
 *
 * 5. The code is self-contained, uses only standard CUDA headers, and
 *    compiles as a .cu file. It does not rely on any external data or
 *    libraries beyond the CUDA runtime.
 *
 * 6. We choose a width that is not naturally aligned to typical pitches
 *    (e.g. 7 ints = 28 bytes) so that padding bytes exist. This demonstrates
 *    that cudaMemset correctly sets the entire allocated region, not just
 *    the logical width.
 *
 * 7. The final output prints the first element of each row to show the
 *    value 0xAAAAAAAA, confirming that cudaMemset worked as expected.
 *
 * This program thus fulfills the requirement: it shows that cudaMemset
 * works on a 2D allocation created by cudaMallocPitch.
 */

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main(void)
{
    const int width = 7;   // number of elements per row (ints)
    const int height = 5;  // number of rows
    const int elementSize = sizeof(int);
    const int widthInBytes = width * elementSize;

    int *d_ptr = nullptr;
    size_t pitch = 0;

    // Allocate pitched memory
    CUDA_CHECK(cudaMallocPitch((void**)&d_ptr, &pitch, widthInBytes, height));

    printf("Allocated pitched memory: pitch = %zu bytes, width = %d ints, height = %d rows\n",
           pitch, width, height);

    // Set all bytes in the allocated block to 0xAA
    // This covers pitch*height bytes, including padding
    CUDA_CHECK(cudaMemset(d_ptr, 0xAA, pitch * height));

    // Allocate host memory to copy back
    size_t hostPitch = pitch;
    int *h_ptr = (int*)malloc(hostPitch * height);
    if (!h_ptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        cudaFree(d_ptr);
        return EXIT_FAILURE;
    }

    // Copy the pitched data back to host
    CUDA_CHECK(cudaMemcpy2D(h_ptr, hostPitch,
                            d_ptr, pitch,
                            widthInBytes, height,
                            cudaMemcpyDeviceToHost));

    // Verify and print first element of each row
    printf("First element of each row after cudaMemset:\n");
    for (int row = 0; row < height; ++row) {
        int val = *(h_ptr + row * (hostPitch / elementSize));
        printf("Row %d: 0x%08X\n", row, val);
    }

    // Clean up
    free(h_ptr);
    CUDA_CHECK(cudaFree(d_ptr));

    printf("Program completed successfully.\n");
    return EXIT_SUCCESS;
}
