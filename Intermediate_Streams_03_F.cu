/*
 * Aim: Plot the total execution time vs. the number of streams used (1, 2, 3, 4...). The benefit usually saturates quickly.
 *
 * Thinking:
 * 1. The goal is to illustrate how using multiple CUDA streams can overlap kernel execution and memory transfers,
 *    thereby reducing overall runtime up to a certain saturation point. We will perform a simple vector addition
 *    kernel over a large array and time the execution when the work is split across different numbers of streams.
 *
 * 2. We will allocate a large array of floats on the device, copy inputs from host to device once per experiment,
 *    then launch the kernel multiple times, each time dividing the array into N chunks, where N is the number of
 *    streams. Each chunk will be processed by a distinct stream.
 *
 * 3. Timing will be performed using CUDA events. For each experiment (given number of streams) we record a start
 *    event before launching all kernels, and an end event after we synchronize on the last stream (or all streams).
 *    The elapsed time in milliseconds is then printed along with the stream count.
 *
 * 4. The output will be a simple CSV-like format: <num_streams>,<time_ms>. This can be redirected to a file and
 *    plotted externally using tools like gnuplot, matplotlib, etc.
 *
 * 5. For simplicity and portability we set the array size to 100 million floats (~400 MB). This is large enough to
 *    show the benefit of overlapping, but may need to be reduced on systems with less memory.
 *
 * 6. We perform a minimal amount of host code per experiment: copying inputs to device, launching kernels, and
 *    measuring time. We reuse device memory across experiments to avoid measuring allocation overhead.
 *
 * 7. Error checking is handled by a macro that prints the error string and exits on failure. This ensures we catch
 *    any issues early.
 *
 * 8. We set a maximum number of streams (e.g., 8) but this can be adjusted by changing MAX_STREAMS. The program
 *    prints timing for stream counts from 1 to MAX_STREAMS inclusive.
 *
 * 9. The kernel itself is a simple element-wise addition: c[i] = a[i] + b[i].
 *
 * 10. The program is self-contained and can be compiled with `nvcc -o stream_benchmark stream_benchmark.cu`.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/* Error checking macro */
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* Kernel: vector addition */
__global__ void vecAdd(const float *a, const float *b, float *c, size_t N, size_t offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t global_idx = offset + idx;
    if (global_idx < N)
        c[global_idx] = a[global_idx] + b[global_idx];
}

/* Parameters */
const size_t ARRAY_SIZE = 100 * 1000 * 1000;   /* 100 million elements (~400MB) */
const size_t BLOCK_SIZE = 256;                 /* Threads per block */
const int   MAX_STREAMS = 8;                   /* Max number of streams to test */

int main(void)
{
    /* Allocate host memory */
    float *h_a = (float *)malloc(ARRAY_SIZE * sizeof(float));
    float *h_b = (float *)malloc(ARRAY_SIZE * sizeof(float));
    float *h_c = (float *)malloc(ARRAY_SIZE * sizeof(float));
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input data */
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(ARRAY_SIZE - i);
    }

    /* Allocate device memory */
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_a, ARRAY_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_b, ARRAY_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_c, ARRAY_SIZE * sizeof(float)));

    /* Copy inputs to device once (they will be reused) */
    CHECK_CUDA(cudaMemcpy(d_a, h_a, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* Header for output */
    printf("streams,time_ms\n");

    /* Loop over number of streams */
    for (int num_streams = 1; num_streams <= MAX_STREAMS; ++num_streams) {
        /* Create streams */
        cudaStream_t *streams = (cudaStream_t *)malloc(num_streams * sizeof(cudaStream_t));
        for (int s = 0; s < num_streams; ++s) {
            CHECK_CUDA(cudaStreamCreate(&streams[s]));
        }

        /* Determine chunk size per stream */
        size_t chunk = (ARRAY_SIZE + num_streams - 1) / num_streams;  /* ceil division */

        /* Record start event */
        CHECK_CUDA(cudaEventRecord(start, 0));

        /* Launch kernels on each stream */
        for (int s = 0; s < num_streams; ++s) {
            size_t offset = s * chunk;
            size_t N_chunk = (offset + chunk <= ARRAY_SIZE) ? chunk : (ARRAY_SIZE - offset);
            size_t threads = BLOCK_SIZE;
            size_t blocks = (N_chunk + threads - 1) / threads;
            if (blocks == 0) continue; /* Skip if no work */
            vecAdd<<<blocks, threads, 0, streams[s]>>>(d_a, d_b, d_c, ARRAY_SIZE, offset);
            CHECK_CUDA(cudaGetLastError());
        }

        /* Record stop event on default stream after all kernels have finished */
        CHECK_CUDA(cudaEventRecord(stop, 0));

        /* Wait for all streams to complete */
        CHECK_CUDA(cudaEventSynchronize(stop));

        /* Calculate elapsed time */
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

        /* Output result */
        printf("%d,%.3f\n", num_streams, elapsed_ms);

        /* Destroy streams */
        for (int s = 0; s < num_streams; ++s) {
            CHECK_CUDA(cudaStreamDestroy(streams[s]));
        }
        free(streams);
    }

    /* Clean up */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return EXIT_SUCCESS;
}
