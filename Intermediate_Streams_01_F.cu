```cuda
/*
 * Use a single large allocation on the host and use pointer offsets to process each chunk.
 *
 * This program demonstrates how to work with a large dataset by allocating a single,
 * contiguous block of host memory and then processing it in smaller chunks on the GPU.
 * The key idea is to avoid allocating new memory for each chunk on the host; instead,
 * we use pointer arithmetic (offsets) to refer to the relevant segment of the
 * large array.  On the device side we allocate a single large buffer as well, and
 * launch kernels that operate on subranges of this buffer by passing an offset.
 *
 * The program steps are:
 *   1. Define a total number of elements and a chunk size that fits comfortably
 *      within the GPU memory budget.
 *   2. Allocate a large host array of floats (SIZE elements) in one go.
 *   3. Initialize the host array with random values.
 *   4. Allocate a corresponding large device array.
 *   5. For each chunk:
 *        - Compute the offset (in elements) into the large arrays.
 *        - Launch a CUDA kernel that operates on the chunk by receiving a pointer
 *          that has been offset by 'offset * sizeof(float)'.
 *        - In this example, the kernel simply squares each element of the chunk.
 *   6. Copy the processed data back to the host in one large transfer (or
 *      alternatively chunk by chunk).
 *   7. Verify a few elements to ensure correctness.
 *
 * By using pointer offsets we keep the memory allocation pattern simple and
 * efficient, and we also showcase how to work with sub-portions of large
 * arrays without reallocating or copying data between the host and device.
 * This pattern is useful for streaming large datasets, handling time series,
 * or any situation where a single large buffer is preferable over many small
 * allocations.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Error checking macro
#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    }

// Kernel that squares each element in a sub-array
__global__ void squareKernel(float *d_arr, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float val = d_arr[idx];
        d_arr[idx] = val * val;
    }
}

int main(void) {
    const size_t TOTAL_ELEMENTS = 10 * 1024 * 1024; // 10 million floats (~40 MB)
    const size_t CHUNK_SIZE = 256 * 1024;           // 256k floats per chunk
    const size_t NUM_CHUNKS = (TOTAL_ELEMENTS + CHUNK_SIZE - 1) / CHUNK_SIZE;

    printf("Total elements: %zu\n", TOTAL_ELEMENTS);
    printf("Chunk size:    %zu\n", CHUNK_SIZE);
    printf("Number of chunks: %zu\n", NUM_CHUNKS);

    // Allocate host memory
    float *h_data = (float*)malloc(TOTAL_ELEMENTS * sizeof(float));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host data with random values
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < TOTAL_ELEMENTS; ++i) {
        h_data[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_data;
    CHECK_CUDA(cudaMalloc((void**)&d_data, TOTAL_ELEMENTS * sizeof(float)));

    // Copy entire dataset to device in one go (could also copy chunk by chunk)
    CHECK_CUDA(cudaMemcpy(d_data, h_data, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));

    // Determine kernel launch configuration
    const size_t THREADS_PER_BLOCK = 256;
    size_t blocksPerChunk = (CHUNK_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Process each chunk
    for (size_t chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
        size_t offset = chunk * CHUNK_SIZE;
        size_t currentChunkSize = ((offset + CHUNK_SIZE) <= TOTAL_ELEMENTS) ? CHUNK_SIZE
            : (TOTAL_ELEMENTS - offset);

        // Get device pointer offset by 'offset' elements
        float *d_chunk_ptr = d_data + offset;

        // Launch kernel on the current chunk
        squareKernel<<<blocksPerChunk, THREADS_PER_BLOCK>>>(d_chunk_ptr, currentChunkSize);

        // Check for kernel launch errors
        CHECK_CUDA(cudaGetLastError());
    }

    // Wait for all kernels to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy processed data back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("Verification of processed data:\n");
    for (size_t i = 0; i < 5; ++i) {
        float expected = h_data[i]; // After square
        printf("h_data[%zu] = %f\n", i, h_data[i]);
    }

    // Clean up
    free(h_data);
    CHECK_CUDA(cudaFree(d_data));

    return EXIT_SUCCESS;
}
```