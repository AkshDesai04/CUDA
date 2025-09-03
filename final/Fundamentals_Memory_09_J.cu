/*
Aim: Create a table in your output showing Pageable HtoD, Pinned HtoD, Pageable DtoH, and Pinned DtoH bandwidths.

Thinking:
The program must measure memory transfer bandwidths between host and device for both pageable and pinned host memory. 
To do this we will:
1. Allocate a large buffer (~512 MB) as pageable memory using malloc, and another of the same size using cudaMallocHost for pinned memory.
2. Allocate a device buffer of the same size with cudaMalloc.
3. For each transfer type (Pageable HtoD, Pinned HtoD, Pageable DtoH, Pinned DtoH):
   - Create cudaEvent_t objects for timing.
   - Record the start event, perform the cudaMemcpy with the appropriate direction and memory type, record the stop event.
   - Synchronize and compute elapsed time in milliseconds.
   - Convert the elapsed time to seconds and calculate bandwidth in GB/s as (bytes / seconds) / 1e9.
4. Print the results in a simple table with clear labels.
5. Clean up all allocated memory and CUDA events.

The code includes error checking via a macro that aborts on any CUDA API failure. 
The buffer size is chosen to be large enough to provide meaningful bandwidth measurements while fitting comfortably in system memory. 
The program is self‑contained and can be compiled with nvcc to produce a .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void) {
    const size_t MB = 1024 * 1024;
    const size_t size_bytes = 512 * MB;            // 512 MB
    const size_t num_floats = size_bytes / sizeof(float);

    // Allocate pageable host memory
    float *pageable_h = (float*)malloc(size_bytes);
    if (!pageable_h) {
        fprintf(stderr, "Failed to allocate pageable host memory.\n");
        return EXIT_FAILURE;
    }

    // Allocate pinned host memory
    float *pinned_h = NULL;
    CHECK_CUDA(cudaMallocHost(&pinned_h, size_bytes));

    // Allocate device memory
    float *d = NULL;
    CHECK_CUDA(cudaMalloc(&d, size_bytes));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float milliseconds = 0.0f;
    double bandwidth_GBps = 0.0;

    double pageable_h2d_bw, pinned_h2d_bw, pageable_d2h_bw, pinned_d2h_bw;

    // Pageable Host to Device
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d, pageable_h, size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    pageable_h2d_bw = (double)size_bytes / (milliseconds * 1e-3) / 1e9;

    // Pinned Host to Device
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d, pinned_h, size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    pinned_h2d_bw = (double)size_bytes / (milliseconds * 1e-3) / 1e9;

    // Pageable Device to Host
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(pageable_h, d, size_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    pageable_d2h_bw = (double)size_bytes / (milliseconds * 1e-3) / 1e9;

    // Pinned Device to Host
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(pinned_h, d, size_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    pinned_d2h_bw = (double)size_bytes / (milliseconds * 1e-3) / 1e9;

    // Output table
    printf("%-15s | %10s\n", "Transfer", "Bandwidth (GB/s)");
    printf("-----------------|------------\n");
    printf("%-15s | %10.2f\n", "Pageable HtoD", pageable_h2d_bw);
    printf("%-15s | %10.2f\n", "Pinned HtoD", pinned_h2d_bw);
    printf("%-15s | %10.2f\n", "Pageable DtoH", pageable_d2h_bw);
    printf("%-15s | %10.2f\n", "Pinned DtoH", pinned_d2h_bw);

    // Cleanup
    free(pageable_h);
    CHECK_CUDA(cudaFreeHost(pinned_h));
    CHECK_CUDA(cudaFree(d));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
