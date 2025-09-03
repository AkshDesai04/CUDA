/*
Use a single large allocation on the host and use pointer offsets to process each chunk.

Thinking:
The goal of this CUDA program is to demonstrate how to work with a large dataset that resides in a single contiguous host memory block while processing it on the GPU in smaller manageable chunks. 
We allocate one large array on the host, initialize it, and then split the processing into sub‑arrays (chunks). 
For each chunk we:
1. Compute a pointer offset into the large host array (`float *h_chunk_ptr = h_data + offset;`).
2. Copy that chunk to a small device buffer (`cudaMemcpy(d_chunk, h_chunk_ptr, ...)`).
3. Launch a kernel that operates on the chunk (here we simply add 1.0f to each element).
4. Copy the results back to the same offset location in the large host array.

By using pointer arithmetic on the host side we avoid allocating many separate host buffers; all data is contiguous and can be accessed via simple offsets. 
The device memory is reused for each chunk, which keeps the GPU memory footprint small. 
This pattern is common when the entire dataset is too large to fit on the GPU or when you want to reuse a fixed amount of device memory for streaming large data. 
The code includes basic error checking and a simple verification step at the end to ensure that the GPU processing produced the expected result.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 20)              // Total number of elements (1,048,576)
#define CHUNK_SIZE (1 << 18)      // Size of each chunk (262,144)
#define THREADS_PER_BLOCK 256

// Simple kernel that adds 1.0f to each element in the chunk
__global__ void add_one_kernel(float *data, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] += 1.0f;
    }
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main(void)
{
    // Allocate single large host array
    float *h_data = (float *)malloc(N * sizeof(float));
    if (!h_data)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (size_t i = 0; i < N; ++i)
    {
        h_data[i] = (float)i;
    }

    // Allocate device buffer for a single chunk
    float *d_chunk;
    CUDA_CHECK(cudaMalloc((void **)&d_chunk, CHUNK_SIZE * sizeof(float)));

    // Process each chunk
    for (size_t offset = 0; offset < N; offset += CHUNK_SIZE)
    {
        size_t chunk_size = (offset + CHUNK_SIZE <= N) ? CHUNK_SIZE : N - offset;
        float *h_chunk_ptr = h_data + offset;  // pointer offset into large host array

        // Copy chunk from host to device
        CUDA_CHECK(cudaMemcpy(d_chunk, h_chunk_ptr, chunk_size * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Launch kernel on the chunk
        size_t num_blocks = (chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        add_one_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_chunk, chunk_size);
        CUDA_CHECK(cudaGetLastError());

        // Copy processed chunk back to host
        CUDA_CHECK(cudaMemcpy(h_chunk_ptr, d_chunk, chunk_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_chunk));

    // Simple verification: each element should now be original + 1
    int errors = 0;
    for (size_t i = 0; i < N; ++i)
    {
        float expected = (float)i + 1.0f;
        if (h_data[i] != expected)
        {
            if (errors < 10) // Print first few errors
            {
                fprintf(stderr, "Error at index %zu: got %f, expected %f\n",
                        i, h_data[i], expected);
            }
            ++errors;
        }
    }

    if (errors == 0)
    {
        printf("All %d elements processed correctly.\n", (int)N);
    }
    else
    {
        printf("Found %d mismatches.\n", errors);
    }

    // Clean up host memory
    free(h_data);

    // Reset device
    CUDA_CHECK(cudaDeviceReset());

    return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
