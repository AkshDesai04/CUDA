```c
/*
 * Free the pitched memory using `cudaFree`.
 *
 * Thinking:
 * 1. The program will demonstrate how to allocate a 2D array in pitched memory
 *    using cudaMallocPitch, perform a simple kernel launch that writes to the
 *    allocated memory, and then free the pitched memory using cudaFree.
 *
 * 2. We will:
 *    - Include <cuda_runtime.h> for CUDA runtime API.
 *    - Define a simple kernel that writes the thread's 2D index to the
 *      pitched array. The kernel will be launched with grid/block dimensions
 *      that cover the array size.
 *    - In main():
 *        a. Set array dimensions (width in bytes, height in elements).
 *        b. Allocate pitched memory using cudaMallocPitch.
 *        c. Allocate a corresponding host array to check results.
 *        d. Launch kernel to fill device memory.
 *        e. Copy data back to host using cudaMemcpy2D.
 *        f. Print a few elements to confirm correct write.
 *        g. Free pitched device memory using cudaFree.
 *        h. Free host memory.
 *
 * 3. The code includes error checking macros for brevity, ensuring that any
 *    CUDA API call failure is reported. The goal is to highlight the use of
 *    cudaFree to clean up pitched memory, which works the same as normal
 *    cudaFree since the memory pointer returned by cudaMallocPitch is a
 *    single device pointer.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",            \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                          \
        }                                                                 \
    } while (0)

// Simple kernel that writes thread indices into pitched 2D array
__global__ void write_pitched(unsigned int *pitch_ptr, size_t pitch, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (x < width && y < height) {
        // Calculate the address considering pitch
        unsigned int *row = (unsigned int*)((char*)pitch_ptr + y * pitch);
        row[x] = y * width + x; // store a unique value per element
    }
}

int main(void) {
    const int width = 256;   // number of columns (elements per row)
    const int height = 128;  // number of rows

    size_t pitch_bytes;           // pitch in bytes
    unsigned int *dev_pitch_ptr;  // device pointer returned by cudaMallocPitch

    // Allocate pitched memory on the device
    CHECK_CUDA(cudaMallocPitch(&dev_pitch_ptr, &pitch_bytes,
                               width * sizeof(unsigned int), height));

    // Set up execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel to write data
    write_pitched<<<gridSize, blockSize>>>(dev_pitch_ptr, pitch_bytes, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Allocate host memory for copy-back
    unsigned int *host_data = (unsigned int*)malloc(pitch_bytes * height);
    if (!host_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Copy 2D pitched memory from device to host
    CHECK_CUDA(cudaMemcpy2D(host_data, pitch_bytes,
                            dev_pitch_ptr, pitch_bytes,
                            width * sizeof(unsigned int), height,
                            cudaMemcpyDeviceToHost));

    // Print a few values to verify
    printf("Sample values from pitched array:\n");
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            printf("%5u ", host_data[y * (pitch_bytes / sizeof(unsigned int)) + x]);
        }
        printf("\n");
    }

    // Free the pitched device memory using cudaFree
    CHECK_CUDA(cudaFree(dev_pitch_ptr));

    // Clean up host memory
    free(host_data);

    printf("Pitched memory freed successfully.\n");
    return EXIT_SUCCESS;
}
```