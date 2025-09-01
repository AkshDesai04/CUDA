/*
Aim: Allocate memory and use `cudaMemsetAsync` with a non-default stream.

Thinking:
1. Allocate a simple device buffer with cudaMalloc.
2. Create a non-default CUDA stream with cudaStreamCreate.
3. Use cudaMemsetAsync to fill the buffer asynchronously in that stream.
4. Copy the data back to host asynchronously with the same stream.
5. Synchronize the stream to ensure all operations are finished.
6. Verify that the host data matches the expected pattern.
7. Add error checking after each CUDA call via a helper macro.
8. Keep the example self‑contained and compilable with nvcc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    const size_t N = 10;                     // Number of integers
    const size_t bytes = N * sizeof(int);    // Total bytes to allocate
    int *d_array = NULL;                     // Device pointer
    int *h_array = NULL;                     // Host pointer
    cudaStream_t stream;                     // Non-default stream

    /* Allocate host memory */
    h_array = (int *)malloc(bytes);
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }
    /* Allocate device memory */
    CHECK_CUDA(cudaMalloc((void **)&d_array, bytes));

    /* Create a non-default stream */
    CHECK_CUDA(cudaStreamCreate(&stream));

    /* Asynchronously set the device memory to a known pattern (0x55) */
    CHECK_CUDA(cudaMemsetAsync(d_array, 0x55, bytes, stream));

    /* Asynchronously copy the device memory back to host */
    CHECK_CUDA(cudaMemcpyAsync(h_array, d_array, bytes,
                               cudaMemcpyDeviceToHost, stream));

    /* Wait for all operations in the stream to complete */
    CHECK_CUDA(cudaStreamSynchronize(stream));

    /* Verify results */
    printf("Host array contents after cudaMemsetAsync and cudaMemcpyAsync:\n");
    for (size_t i = 0; i < N; ++i) {
        printf("h_array[%zu] = 0x%08X (%d)\n", i, h_array[i], h_array[i]);
    }

    /* Clean up */
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_array));
    free(h_array);

    return EXIT_SUCCESS;
}
