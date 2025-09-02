```c
/*
Time both HtoD and DtoH transfers with pinned memory and compare to the pageable case.
The aim of the program is to measure and compare the time taken for copying data
between host and device when using pageable (regular) host memory versus pinned
(host) memory. The program performs the following steps:

1. Define a data size (number of float elements) that is large enough to give
   measurable transfer times but small enough to fit into typical system RAM.
2. Allocate a device array using cudaMalloc.
3. Allocate two host arrays: one using regular malloc (pageable) and one using
   cudaMallocHost (pinned).
4. Initialize the pageable and pinned host arrays with some data.
5. Create CUDA events for timing.
6. Perform a loop of a few iterations (e.g. 10) to average timings.
   In each iteration:
   - Record start event.
   - Copy from pageable host to device (HtoD).
   - Record mid event.
   - Copy from device to pageable host (DtoH).
   - Record end event.
   - Calculate elapsed times for HtoD and DtoH and accumulate.
7. Repeat the same process for the pinned host memory.
8. Compute average transfer times for each case.
9. Print the results, showing the transfer times for pageable and pinned
   host memory for both HtoD and DtoH.
10. Clean up all allocated memory and destroy events.

Error checking macros are used to ensure any CUDA API failure is reported
immediately. The program uses cudaEventElapsedTime which returns time in
milliseconds. The code is self‑contained and can be compiled with nvcc, e.g.:
   nvcc -o transfer_time transfer_time.cu
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10000000          // 10 million floats (~40 MB)
#define NUM_ITER 10

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %d: %s.\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main(void) {
    float *d_array = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_array, N * sizeof(float)));

    // Allocate pageable host memory
    float *h_pageable = (float*)malloc(N * sizeof(float));
    if (h_pageable == NULL) {
        fprintf(stderr, "Failed to allocate pageable host memory.\n");
        exit(EXIT_FAILURE);
    }

    // Allocate pinned host memory
    float *h_pinned = NULL;
    CHECK_CUDA(cudaMallocHost((void**)&h_pinned, N * sizeof(float)));

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_pageable[i] = (float)i;
        h_pinned[i]  = (float)i;
    }

    // Events for timing
    cudaEvent_t start, mid, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&mid));
    CHECK_CUDA(cudaEventCreate(&end));

    float pageable_h2d = 0.0f, pageable_d2h = 0.0f;
    float pinned_h2d   = 0.0f, pinned_d2h   = 0.0f;

    // ===========================
    // Pageable memory transfers
    // ===========================
    for (int i = 0; i < NUM_ITER; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));

        // HtoD
        CHECK_CUDA(cudaMemcpy(d_array, h_pageable, N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(mid, 0));

        // DtoH
        CHECK_CUDA(cudaMemcpy(h_pageable, d_array, N * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(end, 0));

        CHECK_CUDA(cudaEventSynchronize(end));

        float ms_h2d, ms_d2h;
        CHECK_CUDA(cudaEventElapsedTime(&ms_h2d, start, mid));
        CHECK_CUDA(cudaEventElapsedTime(&ms_d2h, mid, end));

        pageable_h2d += ms_h2d;
        pageable_d2h += ms_d2h;
    }

    // ===========================
    // Pinned memory transfers
    // ===========================
    for (int i = 0; i < NUM_ITER; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));

        // HtoD
        CHECK_CUDA(cudaMemcpy(d_array, h_pinned, N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(mid, 0));

        // DtoH
        CHECK_CUDA(cudaMemcpy(h_pinned, d_array, N * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(end, 0));

        CHECK_CUDA(cudaEventSynchronize(end));

        float ms_h2d, ms_d2h;
        CHECK_CUDA(cudaEventElapsedTime(&ms_h2d, start, mid));
        CHECK_CUDA(cudaEventElapsedTime(&ms_d2h, mid, end));

        pinned_h2d += ms_h2d;
        pinned_d2h += ms_d2h;
    }

    // Compute averages
    pageable_h2d /= NUM_ITER;
    pageable_d2h /= NUM_ITER;
    pinned_h2d   /= NUM_ITER;
    pinned_d2h   /= NUM_ITER;

    // Output results
    printf("Transfer times over %d iterations (average in ms):\n", NUM_ITER);
    printf("Pageable HtoD: %.3f ms\n", pageable_h2d);
    printf("Pageable DtoH: %.3f ms\n", pageable_d2h);
    printf("Pinned   HtoD: %.3f ms\n", pinned_h2d);
    printf("Pinned   DtoH: %.3f ms\n", pinned_d2h);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(mid));
    CHECK_CUDA(cudaEventDestroy(end));

    free(h_pageable);
    CHECK_CUDA(cudaFreeHost(h_pinned));
    CHECK_CUDA(cudaFree(d_array));

    return 0;
}
```