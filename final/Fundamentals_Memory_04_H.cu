/*
 * Aim of the program: Allocate memory and use `cudaMemsetAsync` with a non-default stream.
 *
 * Thinking:
 * 1. The requirement is to demonstrate allocating device memory and performing
 *    a memset operation asynchronously on a user‑defined CUDA stream.
 * 2. The simplest approach is to allocate a block of integers with cudaMalloc.
 * 3. Create a non‑default stream with cudaStreamCreate.
 * 4. Use cudaMemsetAsync to write a byte value into the allocated memory.
 *    Because cudaMemset operates on bytes, setting all bytes to a value like
 *    0xAA will produce a predictable pattern in each integer (0xAAAAAAAA).
 * 5. Copy the data back to host using cudaMemcpyAsync on the same stream,
 *    then synchronize the stream to ensure all operations are finished.
 * 6. Verify the contents by printing a few elements.
 * 7. Finally, clean up the stream and free the device memory.
 * 8. All API calls are wrapped with error checking for robustness.
 *
 * This program serves as a minimal, self‑contained example that fulfills the
 * requested behavior: allocating memory, using cudaMemsetAsync, and using a
 * non‑default stream. It is written in standard CUDA C and can be compiled
 * with nvcc: nvcc -arch=sm_70 -o memsets_async_example memsets_async_example.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for checking CUDA API errors */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main(void)
{
    const size_t N = 16;                     // Number of integers
    const size_t bytes = N * sizeof(int);    // Size in bytes

    // Allocate device memory
    int *d_array = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_array, bytes));

    // Create a non‑default stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Asynchronously set all bytes to 0xAA in the device array
    // Value is interpreted as a byte, so each int becomes 0xAAAAAAAA
    CUDA_CHECK(cudaMemsetAsync(d_array, 0xAA, bytes, stream));

    // Allocate host memory to copy back the result
    int h_array[N] = {0};

    // Asynchronously copy the data back to host
    CUDA_CHECK(cudaMemcpyAsync(h_array, d_array, bytes,
                               cudaMemcpyDeviceToHost, stream));

    // Wait for the stream to finish
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Print the results
    printf("Device memory contents after cudaMemsetAsync (byte 0xAA):\n");
    for (size_t i = 0; i < N; ++i) {
        printf("h_array[%zu] = 0x%08X\n", i, h_array[i]);
    }

    // Clean up
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_array));

    return 0;
}
