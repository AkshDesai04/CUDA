```cpp
/*
Aim of the program:
Extend the pipeline to use 3 streams and 3 buffers. Does this provide further speedup over 2 streams?

Thinking process:
The original pipeline likely used double buffering with two streams to overlap host‑to‑device copy, kernel execution,
and device‑to‑host copy. To explore further overlap we extend this to three streams and three device/host buffers.
Each stream will handle one chunk of the data, allowing three chunks to be in-flight concurrently. The key idea
is to cycle through the buffers: chunk i is mapped to buffer i % 3 and stream i % 3. Because each stream processes
its buffer sequentially, we avoid conflicts; the streams themselves allow overlapping of the three stages.
We also compare the execution time against a 2‑stream version to see if the extra stream yields a measurable
speedup. The program allocates a large array, splits it into chunks, runs a simple kernel (adding 1.0f), and times
the whole pipeline with cuda events. Error checking is minimal for brevity but is included for key CUDA calls.
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define N 104857600          // Total number of elements (≈400 MB)
#define NUM_CHUNKS 12        // Number of chunks to process
#define CHUNK_SIZE (N / NUM_CHUNKS)

__global__ void addKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1.0f;
}

void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Allocate host data for 3-stream test
    float *h_bufs[3];
    for (int i = 0; i < 3; ++i) {
        cudaMallocHost((void**)&h_bufs[i], CHUNK_SIZE * sizeof(float));
        // Initialize host buffer with some values
        for (int j = 0; j < CHUNK_SIZE; ++j) {
            h_bufs[i][j] = (float)(i * CHUNK_SIZE + j);
        }
    }

    // Allocate device data for 3-stream test
    float *d_bufs[3];
    for (int i = 0; i < 3; ++i) {
        cudaMalloc((void**)&d_bufs[i], CHUNK_SIZE * sizeof(float));
    }

    // Create 3 streams
    cudaStream_t streams[3];
    for (int i = 0; i < 3; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Events for timing
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    // Launch 3-stream pipeline
    cudaEventRecord(start3, 0);
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        int bufIdx = i % 3;
        int streamIdx = i % 3;
        // H2D copy
        cudaMemcpyAsync(d_bufs[bufIdx], h_bufs[bufIdx],
                        CHUNK_SIZE * sizeof(float),
                        cudaMemcpyHostToDevice,
                        streams[streamIdx]);

        // Kernel launch
        int threads = 256;
        int blocks = (CHUNK_SIZE + threads - 1) / threads;
        addKernel<<<blocks, threads, 0, streams[streamIdx]>>>(d_bufs[bufIdx], CHUNK_SIZE);
        checkCudaError("Kernel launch");

        // D2H copy
        cudaMemcpyAsync(h_bufs[bufIdx], d_bufs[bufIdx],
                        CHUNK_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        streams[streamIdx]);
    }

    // Wait for all streams to finish
    for (int i = 0; i < 3; ++i) {
        cudaStreamSynchronize(streams[i]);
    }
    cudaEventRecord(stop3, 0);
    cudaEventSynchronize(stop3);
    float ms3 = 0;
    cudaEventElapsedTime(&ms3, start3, stop3);
    printf("3-stream pipeline time: %f ms\n", ms3);

    // Clean up 3-stream resources
    for (int i = 0; i < 3; ++i) {
        cudaFreeHost(h_bufs[i]);
        cudaFree(d_bufs[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);

    // -----------------------------------------
    // 2-stream test for comparison
    // -----------------------------------------

    // Allocate host data for 2-stream test
    float *h_bufs2[2];
    for (int i = 0; i < 2; ++i) {
        cudaMallocHost((void**)&h_bufs2[i], CHUNK_SIZE * sizeof(float));
        for (int j = 0; j < CHUNK_SIZE; ++j) {
            h_bufs2[i][j] = (float)(i * CHUNK_SIZE + j);
        }
    }

    // Allocate device data for 2-stream test
    float *d_bufs2[2];
    for (int i = 0; i < 2; ++i) {
        cudaMalloc((void**)&d_bufs2[i], CHUNK_SIZE * sizeof(float));
    }

    // Create 2 streams
    cudaStream_t streams2[2];
    for (int i = 0; i < 2; ++i) {
        cudaStreamCreate(&streams2[i]);
    }

    // Events for timing
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    // Launch 2-stream pipeline
    cudaEventRecord(start2, 0);
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        int bufIdx = i % 2;
        int streamIdx = i % 2;
        // H2D copy
        cudaMemcpyAsync(d_bufs2[bufIdx], h_bufs2[bufIdx],
                        CHUNK_SIZE * sizeof(float),
                        cudaMemcpyHostToDevice,
                        streams2[streamIdx]);

        // Kernel launch
        int threads = 256;
        int blocks = (CHUNK_SIZE + threads - 1) / threads;
        addKernel<<<blocks, threads, 0, streams2[streamIdx]>>>(d_bufs2[bufIdx], CHUNK_SIZE);
        checkCudaError("Kernel launch 2-stream");

        // D2H copy
        cudaMemcpyAsync(h_bufs2[bufIdx], d_bufs2[bufIdx],
                        CHUNK_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        streams2[streamIdx]);
    }

    // Wait for all streams to finish
    for (int i = 0; i < 2; ++i) {
        cudaStreamSynchronize(streams2[i]);
    }
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    float ms2 = 0;
    cudaEventElapsedTime(&ms2, start2, stop2);
    printf("2-stream pipeline time: %f ms\n", ms2);

    // Clean up 2-stream resources
    for (int i = 0; i < 2; ++i) {
        cudaFreeHost(h_bufs2[i]);
        cudaFree(d_bufs2[i]);
        cudaStreamDestroy(streams2[i]);
    }
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    // Final comparison
    if (ms3 < ms2) {
        printf("3 streams provided a speedup over 2 streams.\n");
    } else {
        printf("3 streams did not provide a speedup over 2 streams.\n");
    }

    return 0;
}
```