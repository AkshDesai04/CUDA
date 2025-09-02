/*
Aim: Perform a round trip: HtoD, then DtoH. Time the entire process.

Thinking:
To meet the goal, I need a self‑contained CUDA C program that:
1. Allocates a sizable array on the host (CPU) memory.
2. Allocates a matching array on the device (GPU) memory.
3. Copies the data from host to device (HtoD).
4. Copies it back from device to host (DtoH).
5. Times the combined duration of the two memcpy operations.

The CUDA runtime API offers `cudaEvent_t` objects which provide nanosecond resolution
timing suitable for short kernel launches or memory copies. I will create two events
(`start` and `stop`), record them immediately before the first copy and right after
the second copy, then compute the elapsed time in milliseconds.

For error checking I’ll use a simple macro `CHECK_CUDA` that prints the error string
and aborts if any CUDA call fails. The program will also perform a simple data
validation step: after the round‑trip, it will verify that the host array matches
the original contents.

Finally, I’ll output the measured time and a success/failure message.

The code is written in plain C with CUDA extensions and is ready to compile
with `nvcc`. The target is a single .cu file with no external dependencies other
than the CUDA runtime.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors */
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main(void) {
    /* Parameters */
    const size_t N = 100'000'000;          // Number of elements
    const size_t SIZE = N * sizeof(int);   // Size in bytes

    /* Host memory allocation */
    int *h_data = (int *)malloc(SIZE);
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host array with some values */
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = (int)i;
    }

    /* Device memory allocation */
    int *d_data = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_data, SIZE));

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* Record start event */
    CHECK_CUDA(cudaEventRecord(start, 0));

    /* HtoD copy */
    CHECK_CUDA(cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice));

    /* DtoH copy */
    CHECK_CUDA(cudaMemcpy(h_data, d_data, SIZE, cudaMemcpyDeviceToHost));

    /* Record stop event */
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    /* Calculate elapsed time */
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    /* Validate data after round trip */
    int mismatches = 0;
    for (size_t i = 0; i < N; ++i) {
        if (h_data[i] != (int)i) {
            mismatches++;
            if (mismatches <= 10) {
                fprintf(stderr, "Data mismatch at index %zu: expected %d, got %d\n",
                        i, (int)i, h_data[i]);
            }
        }
    }

    if (mismatches == 0) {
        printf("Round trip successful. Time elapsed: %.3f ms\n", ms);
    } else {
        printf("Round trip completed with %d mismatches. Time elapsed: %.3f ms\n",
               mismatches, ms);
    }

    /* Clean up */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);

    return EXIT_SUCCESS;
}
