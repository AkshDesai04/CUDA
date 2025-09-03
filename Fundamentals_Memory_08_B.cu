```cpp
/*
Perform the same timing for a device-to-device copy.

This program benchmarks the time taken to copy data between two memory regions
located on the GPU.  It allocates two device buffers of the same size,
initializes the source buffer (either by copying from the host or by setting
values on the device), and then measures the elapsed time for a device-to-device
cudaMemcpy operation using CUDA events.  The benchmark is repeated multiple
times to obtain an average duration, which helps to smooth out any outliers
caused by transient system activity.

The steps are:
1. Define buffer size and number of repetitions.
2. Allocate device memory for source and destination.
3. Initialize the source buffer (copy from host for simplicity).
4. Create CUDA events for timing.
5. In a loop, record the start event, perform cudaMemcpy with
   cudaMemcpyDeviceToDevice, record the stop event, synchronize, and
   accumulate elapsed time.
6. After the loop, compute the average time in milliseconds and print it.
7. Clean up: destroy events and free device memory.

Error handling is done via a macro that checks CUDA API return codes
and aborts the program on failure, printing an informative message.
This keeps the main code readable while still catching any runtime
issues.  The program is self-contained and can be compiled with nvcc:
    nvcc -o dtd_copy_timing dtd_copy_timing.cu
Then executed directly.

Because the copy is performed entirely on the device, no host-to-device or
device-to-host transfers are involved after the initial source buffer
initialization.  This isolates the timing to the GPU-to-GPU copy path,
providing a clear measurement of that operation's performance.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro to check CUDA errors
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

int main(void) {
    // Parameters
    const size_t num_elements = 1 << 24; // ~16M elements (~64MB for float)
    const size_t num_bytes = num_elements * sizeof(float);
    const int  repetitions   = 10;       // number of times to copy for averaging

    printf("Device-to-device copy benchmark\n");
    printf("Buffer size: %zu bytes (%zu elements)\n", num_bytes, num_elements);
    printf("Repetitions: %d\n", repetitions);
    printf("\n");

    // Allocate host buffer and initialize
    float *h_buf = (float*)malloc(num_bytes);
    if (!h_buf) {
        fprintf(stderr, "Failed to allocate host buffer\n");
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < num_elements; ++i) {
        h_buf[i] = static_cast<float>(i);
    }

    // Allocate device buffers
    float *d_src = NULL;
    float *d_dst = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_src, num_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_dst, num_bytes));

    // Copy data from host to source device buffer
    CHECK_CUDA(cudaMemcpy(d_src, h_buf, num_bytes, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up copy (optional but useful to avoid first-run overhead)
    CHECK_CUDA(cudaMemcpy(d_dst, d_src, num_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing loop
    float elapsed_ms_total = 0.0f;
    for (int i = 0; i < repetitions; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUDA(cudaMemcpy(d_dst, d_src, num_bytes, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        elapsed_ms_total += elapsed_ms;

        printf("Iteration %2d: %8.3f ms\n", i + 1, elapsed_ms);
    }

    float avg_ms = elapsed_ms_total / repetitions;
    printf("\nAverage device-to-device copy time: %8.3f ms\n", avg_ms);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    free(h_buf);

    return 0;
}
```