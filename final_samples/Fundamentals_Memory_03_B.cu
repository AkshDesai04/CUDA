```cpp
/*
 * Time the `cudaMemcpyDeviceToDevice` operation. Compare this timing to a DtoH followed by an HtoD copy. The DtoD copy should be much faster.
 *
 * Thinking:
 * - Allocate two large device buffers (e.g., 100 MB) and a host buffer of the same size.
 * - Fill the source device buffer with some data (cudaMemset or cudaMemcpy from host).
 * - Measure the time of a direct device-to-device copy using cudaEventRecord and cudaEventElapsedTime.
 *   This uses cudaMemcpy with the copy kind cudaMemcpyDeviceToDevice.
 * - Measure the time of a device-to-host followed by a host-to-device copy.
 *   This uses two calls to cudaMemcpy: cudaMemcpyDeviceToHost and cudaMemcpyHostToDevice.
 * - For fairness, the same amount of data is moved in both cases, and the host buffer is reused.
 * - The events automatically synchronize with the device, so we don't need explicit cudaDeviceSynchronize().
 * - After timing, print the elapsed times in milliseconds and the ratio between the two.
 * - Add simple error checking after each CUDA API call for robustness.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void)
{
    const size_t N = 100 * 1024 * 1024; // 100 MB
    const size_t bytes = N * sizeof(float);

    // Allocate device memory
    float *d_src = nullptr;
    float *d_dst = nullptr;
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));

    // Allocate host memory
    float *h_buf = (float*)malloc(bytes);
    if (!h_buf) {
        fprintf(stderr, "Failed to allocate host buffer\n");
        return EXIT_FAILURE;
    }

    // Initialize host buffer with some data and copy to device source buffer
    for (size_t i = 0; i < N; ++i) {
        h_buf[i] = static_cast<float>(i);
    }
    CHECK_CUDA(cudaMemcpy(d_src, h_buf, bytes, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float ms;

    // ---------- Device-to-Device copy ----------
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("Device-to-Device copy time: %f ms\n", ms);
    float dtod_time = ms;

    // ---------- DtoH followed by HtoD ----------
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(h_buf, d_src, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(d_dst, h_buf, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("DtoH + HtoD copy time: %f ms\n", ms);
    float dh_dtod_time = ms;

    printf("Ratio (DtoH+HtoD) / (DtoD): %.2f\n", dh_dtod_time / dtod_time);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    free(h_buf);

    return 0;
}
```