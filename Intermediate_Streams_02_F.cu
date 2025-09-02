/*
 * Aim of the program:
 * Remember to destroy the streams with `cudaStreamDestroy` at the end.
 *
 * My thinking while writing this code:
 * 1. I want to illustrate how to create multiple CUDA streams,
 *    launch kernels on each stream, and then properly destroy them.
 * 2. A simple vector addition kernel will suffice to demonstrate
 *    stream usage. We'll launch the kernel on several streams
 *    to show concurrent execution (assuming the device supports it).
 * 3. After launching the kernels, we need to synchronize on each
 *    stream or on the device to ensure completion before we
 *    copy data back and free resources.
 * 4. Finally, per the requirement, each stream will be destroyed
 *    with `cudaStreamDestroy`. This is important to avoid
 *    resource leaks when streams are created dynamically.
 * 5. I will include error checking for CUDA API calls to
 *    make the code robust and easier to debug.
 * 6. The code is written in C-style CUDA (using .cu extension)
 *    and compiles with `nvcc`. It uses basic standard libraries.
 * 7. The program will run on a CUDA-capable device; if no GPU
 *    is present, it will exit with an error message.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024            // Number of elements in each vector
#define NUM_STREAMS 4     // Number of concurrent streams

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

// Helper macro for error checking
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void)
{
    // Host memory allocation
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Divide data into NUM_STREAMS chunks
    int chunkSize = N / NUM_STREAMS;
    int remainder = N % NUM_STREAMS;

    // Copy data to device in chunks, launch kernels, and store results
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * chunkSize;
        int size = chunkSize + (i == NUM_STREAMS - 1 ? remainder : 0); // last stream gets remainder

        // Asynchronously copy subarrays to device
        CHECK_CUDA(cudaMemcpyAsync(d_A + offset, h_A + offset, size * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_B + offset, h_B + offset, size * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[i]]);

        // Launch kernel on this stream
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_A + offset,
                                                                    d_B + offset,
                                                                    d_C + offset,
                                                                    size);
        // Check for kernel launch errors
        CHECK_CUDA(cudaGetLastError());
    }

    // Copy result back to host asynchronously
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * chunkSize;
        int size = chunkSize + (i == NUM_STREAMS - 1 ? remainder : 0);
        CHECK_CUDA(cudaMemcpyAsync(h_C + offset, d_C + offset, size * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[i]));
    }

    // Synchronize all streams to ensure all operations are complete
    CHECK_CUDA(cudaDeviceSynchronize());

    // Validate results
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            if (errors < 10) { // print first few errors
                printf("Error at index %d: expected %f, got %f\n", i, expected, h_C[i]);
            }
            ++errors;
        }
    }
    if (errors == 0) {
        printf("Vector addition succeeded. All results correct.\n");
    } else {
        printf("Vector addition completed with %d errors.\n", errors);
    }

    // Destroy streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset device (optional)
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
