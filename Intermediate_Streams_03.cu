```c
/*
Aim: Measure the performance improvement of the streamed version over the sequential version. 
The overlap of copy and execution should result in a significant speedup, hiding the data transfer latency.

Thought process:
- Use a simple vector addition kernel to demonstrate overlap between memory transfer and kernel execution.
- Allocate pinned host memory to enable asynchronous copies with cudaMemcpyAsync.
- Use multiple CUDA streams (NUM_STREAMS) to divide the work into chunks that can be processed concurrently.
- For the sequential version, perform a single copy of the entire data, launch the kernel, and copy back the result.
- For the streamed version, split the data into equal-sized chunks, and for each chunk:
    * Asynchronously copy the chunk from host to device.
    * Launch the kernel on that chunk.
    * Asynchronously copy the result back to host.
  All operations for a chunk are issued on the same stream to enforce correct ordering within the chunk.
- Use cudaEventRecord and cudaEventElapsedTime to time both approaches accurately.
- Verify correctness by comparing the sum of elements in the results from both versions.
- Compute and print the speedup achieved by the streamed version.
- Include error checking after each CUDA API call for robustness.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",  \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                          \
    } while (0)

// Kernel: simple vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 24;            // Total number of elements (16M)
    const int NUM_STREAMS = 4;        // Number of streams for overlapped execution
    const int CHUNK = N / NUM_STREAMS; // Elements per stream (assumes divisible)
    const int THREADS_PER_BLOCK = 256;
    const int BLOCKS_PER_GRID = (CHUNK + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("Vector size: %d elements\n", N);
    printf("Number of streams: %d\n", NUM_STREAMS);
    printf("Chunk size per stream: %d elements\n", CHUNK);

    // Allocate pinned host memory
    float *hA, *hB, *hC_seq, *hC_stream;
    CUDA_CHECK(cudaMallocHost((void**)&hA, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&hB, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&hC_seq, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&hC_stream, N * sizeof(float)));

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        hA[i] = (float)i;
        hB[i] = (float)(N - i);
    }

    // Device memory for sequential version
    float *dA_seq, *dB_seq, *dC_seq;
    CUDA_CHECK(cudaMalloc((void**)&dA_seq, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dB_seq, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dC_seq, N * sizeof(float)));

    // Device memory for streamed version
    float *dA_stream[NUM_STREAMS];
    float *dB_stream[NUM_STREAMS];
    float *dC_stream[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaMalloc((void**)&dA_stream[i], CHUNK * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&dB_stream[i], CHUNK * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&dC_stream[i], CHUNK * sizeof(float)));
    }

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Events for timing
    cudaEvent_t start_seq, stop_seq;
    cudaEvent_t start_stream, stop_stream;
    CUDA_CHECK(cudaEventCreate(&start_seq));
    CUDA_CHECK(cudaEventCreate(&stop_seq));
    CUDA_CHECK(cudaEventCreate(&start_stream));
    CUDA_CHECK(cudaEventCreate(&stop_stream));

    // ---------------------- Sequential Execution ----------------------
    CUDA_CHECK(cudaEventRecord(start_seq, 0));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(dA_seq, hA, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_seq, hB, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    vectorAdd<<<(N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dA_seq, dB_seq, dC_seq, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(hC_seq, dC_seq, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop_seq, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_seq));

    float ms_seq = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_seq, start_seq, stop_seq));
    printf("Sequential time: %.3f ms\n", ms_seq);

    // ---------------------- Streamed Execution ----------------------
    CUDA_CHECK(cudaEventRecord(start_stream, 0));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * CHUNK;
        size_t bytes = CHUNK * sizeof(float);

        // Asynchronously copy chunk to device
        CUDA_CHECK(cudaMemcpyAsync(dA_stream[i], hA + offset, bytes, cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(dB_stream[i], hB + offset, bytes, cudaMemcpyHostToDevice, streams[i]));

        // Launch kernel on the stream
        vectorAdd<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, 0, streams[i]>>>(dA_stream[i], dB_stream[i], dC_stream[i], CHUNK);
        CUDA_CHECK(cudaGetLastError());

        // Asynchronously copy result back
        CUDA_CHECK(cudaMemcpyAsync(hC_stream + offset, dC_stream[i], bytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    // Wait for all streams to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop_stream, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_stream));

    float ms_stream = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_stream, start_stream, stop_stream));
    printf("Streamed time: %.3f ms\n", ms_stream);

    // ---------------------- Validation ----------------------
    double sum_seq = 0.0, sum_stream = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_seq += hC_seq[i];
        sum_stream += hC_stream[i];
    }
    printf("Sum of sequential result: %.2f\n", sum_seq);
    printf("Sum of streamed result:   %.2f\n", sum_stream);
    if (fabs(sum_seq - sum_stream) < 1e-3) {
        printf("Validation PASSED: Results match.\n");
    } else {
        printf("Validation FAILED: Results differ.\n");
    }

    // ---------------------- Speedup ----------------------
    double speedup = ms_seq / ms_stream;
    printf("Speedup (sequential / streamed): %.2fx\n", speedup);

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaFree(dA_stream[i]));
        CUDA_CHECK(cudaFree(dB_stream[i]));
        CUDA_CHECK(cudaFree(dC_stream[i]));
    }
    CUDA_CHECK(cudaFree(dA_seq));
    CUDA_CHECK(cudaFree(dB_seq));
    CUDA_CHECK(cudaFree(dC_seq));

    CUDA_CHECK(cudaFreeHost(hA));
    CUDA_CHECK(cudaFreeHost(hB));
    CUDA_CHECK(cudaFreeHost(hC_seq));
    CUDA_CHECK(cudaFreeHost(hC_stream));

    CUDA_CHECK(cudaEventDestroy(start_seq));
    CUDA_CHECK(cudaEventDestroy(stop_seq));
    CUDA_CHECK(cudaEventDestroy(start_stream));
    CUDA_CHECK(cudaEventDestroy(stop_stream));

    return 0;
}
```