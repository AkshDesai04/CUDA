/*
Aim of the program (word to word as requested):
Write a program that processes a large dataset by breaking it into N chunks. Perform the following sequence N times in a loop on the default stream: (1) Copy HtoD, (2) Execute Kernel, (3) Copy DtoH. Time the total operation.

Thinking:
1. The program will be written in CUDA C (.cu) and compiled with nvcc.
2. A large host dataset (float array) will be allocated and initialized with arbitrary values (e.g., random floats).
3. The dataset will be divided into chunks of a specified size (chunkSize). The number of chunks, N, is computed as ceil(totalSize / chunkSize).
4. For each chunk, the following steps are performed on the default stream:
   a. Copy the chunk from host to device (cudaMemcpy HtoD).
   b. Launch a simple kernel that operates on the chunk. For demonstration, the kernel will add a constant value (e.g., 1.0f) to each element.
   c. Copy the processed chunk back from device to host (cudaMemcpy DtoH).
5. CUDA events (cudaEvent_t) are used to time the entire loop (start before the loop, stop after the loop, and compute elapsed time in milliseconds).
6. The program outputs the elapsed time and optionally verifies a few elements to ensure correctness.
7. Memory is freed and CUDA events are destroyed before exiting.
8. The code is self‑contained and can be compiled with:
      nvcc -o chunk_process chunk_process.cu
9. The program uses default stream implicitly; no explicit stream creation is needed.
10. Edge case: If the total size is not a multiple of the chunk size, the last chunk is processed with its actual size.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),    \
                    cudaGetErrorString(err));                          \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Simple kernel: add a constant value to each element of the array
__global__ void addConstKernel(float *d_data, float value, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
        d_data[idx] += value;
}

int main(int argc, char *argv[])
{
    // Parameters (can be overridden by command line arguments)
    size_t totalSize = 1 << 26;   // Total number of floats (~67 million, ~268 MB)
    size_t chunkSize = 1 << 20;   // Chunk size in floats (~1 million, ~4 MB)
    float addValue = 1.0f;        // Value to add in kernel

    if (argc >= 2) {
        totalSize = strtoull(argv[1], NULL, 10);
    }
    if (argc >= 3) {
        chunkSize = strtoull(argv[2], NULL, 10);
    }

    size_t N = (totalSize + chunkSize - 1) / chunkSize; // Number of chunks

    printf("Total elements: %zu, Chunk size: %zu, Number of chunks: %zu\n",
           totalSize, chunkSize, N);

    // Allocate host memory
    float *h_input  = (float *)malloc(totalSize * sizeof(float));
    float *h_output = (float *)malloc(totalSize * sizeof(float));
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input with random data
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < totalSize; ++i) {
        h_input[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory for one chunk
    float *d_chunk = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_chunk, chunkSize * sizeof(float)));

    // CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Process each chunk
    for (size_t chunkIdx = 0; chunkIdx < N; ++chunkIdx) {
        size_t offset = chunkIdx * chunkSize;
        size_t currentChunkSize = ((offset + chunkSize) <= totalSize) ? chunkSize
                                                                    : (totalSize - offset);

        // Copy current chunk HtoD
        CHECK_CUDA(cudaMemcpy(d_chunk, h_input + offset,
                              currentChunkSize * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Determine grid size
        int threadsPerBlock = 256;
        int blocksPerGrid = (currentChunkSize + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel on default stream
        addConstKernel<<<blocksPerGrid, threadsPerBlock>>>(d_chunk, addValue, (int)currentChunkSize);
        CHECK_CUDA(cudaGetLastError()); // Check kernel launch errors

        // Copy result back DtoH
        CHECK_CUDA(cudaMemcpy(h_output + offset, d_chunk,
                              currentChunkSize * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Total elapsed time: %.3f ms\n", milliseconds);

    // Optional verification: print first 5 elements
    printf("First 5 elements of output:\n");
    for (int i = 0; i < 5; ++i) {
        printf("%.5f ", h_output[i]);
    }
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaFree(d_chunk));
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
