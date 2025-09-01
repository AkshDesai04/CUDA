/*
Aim: Perform a round trip: HtoD, then DtoH. Time the entire process.

Thinking:
To measure the time for transferring data from host to device (HtoD) and back from device to host (DtoH), we will use CUDA events because they provide high-resolution timing for GPU operations. The procedure will be:
1. Allocate a buffer on the host (float array) and initialize it with some data.
2. Allocate a corresponding buffer on the device.
3. Create two CUDA events: start and stop.
4. Record the start event, copy the buffer from host to device (cudaMemcpyHostToDevice), then copy it back (cudaMemcpyDeviceToHost), and record the stop event.
5. Synchronize and compute the elapsed time between the two events.
6. Print the elapsed time in milliseconds.
7. Clean up all allocated memory and destroy the events.

We will include a small error-checking macro to simplify CUDA API error handling. The code is written in CUDA C and can be compiled with `nvcc`. It is self-contained and can be executed directly after compilation. The program demonstrates the round-trip transfer latency and can be used as a benchmark for memory copy performance.

*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple macro for checking CUDA errors */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main(void) {
    const size_t N = 1 << 20;          // 1M floats (~4 MB)
    const size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_buf = (float *)malloc(bytes);
    if (!h_buf) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host buffer */
    for (size_t i = 0; i < N; ++i) {
        h_buf[i] = (float)i;
    }

    /* Allocate device memory */
    float *d_buf;
    CUDA_CHECK(cudaMalloc((void **)&d_buf, bytes));

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Record start event */
    CUDA_CHECK(cudaEventRecord(start, 0));

    /* HtoD copy */
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));

    /* DtoH copy */
    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));

    /* Record stop event */
    CUDA_CHECK(cudaEventRecord(stop, 0));

    /* Wait for the stop event to complete */
    CUDA_CHECK(cudaEventSynchronize(stop));

    /* Calculate elapsed time */
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("Round-trip HtoD + DtoH time: %f ms\n", ms);

    /* Clean up */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_buf));
    free(h_buf);

    return EXIT_SUCCESS;
}
