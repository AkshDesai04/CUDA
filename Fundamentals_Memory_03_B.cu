```cuda
/*********************************************************************
 * Aim of the program: Time the `cudaMemcpyDeviceToDevice` operation.
 * Compare this timing to a DtoH followed by an HtoD copy.
 * The DtoD copy should be much faster.
 *
 * Thinking:
 * 1. Allocate two device arrays (`d_src` and `d_dst`) of the same size.
 * 2. Allocate a host array (`h_src`) to initialize the device source
 *    with known data (e.g., a simple arithmetic sequence).
 * 3. Copy the host data to `d_src` once (setup).
 * 4. Measure the time for a device-to-device copy (`d_src -> d_dst`) using
 *    CUDA events. Repeat this many times (e.g., 100 iterations) and
 *    compute the average time.
 * 5. Measure the time for a host-to-device-to-device path:
 *    * Copy `d_src -> h_src` (device to host).
 *    * Copy `h_src -> d_dst` (host to device).
 *    Use CUDA events for timing and average over the same number of
 *    iterations.
 * 6. Print the average times for both approaches and comment that
 *    the DtoD path should be significantly faster because it stays
 *    entirely on the GPU, avoiding PCIe traffic.
 *
 * Note:
 * - We use `cudaEventElapsedTime` which returns milliseconds.
 * - Error checking is performed after each CUDA call.
 * - The program is self‑contained and compiles with `nvcc`.
 *********************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA launch or API call
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

int main() {
    const size_t N = 1 << 20;          // 1M elements
    const size_t SIZE_BYTES = N * sizeof(float);

    // Allocate host memory
    float *h_src = (float*)malloc(SIZE_BYTES);
    if (!h_src) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_src[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_src = nullptr;
    float *d_dst = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_src, SIZE_BYTES));
    CHECK_CUDA(cudaMalloc((void**)&d_dst, SIZE_BYTES));

    // Copy host data to device source once
    CHECK_CUDA(cudaMemcpy(d_src, h_src, SIZE_BYTES, cudaMemcpyHostToDevice));

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_ITER = 100;
    float ms_per_dtoD = 0.0f;
    float ms_per_dtoH_htoD = 0.0f;

    // Warm-up (optional)
    CHECK_CUDA(cudaMemcpy(d_dst, d_src, SIZE_BYTES, cudaMemcpyDeviceToDevice));

    // ----------------------------
    // Timing Device-to-Device copy
    // ----------------------------
    for (int i = 0; i < NUM_ITER; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUDA(cudaMemcpy(d_dst, d_src, SIZE_BYTES, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms_per_dtoD += ms;
    }
    ms_per_dtoD /= NUM_ITER;

    // ----------------------------------------------
    // Timing Device-to-Host then Host-to-Device copy
    // ----------------------------------------------
    for (int i = 0; i < NUM_ITER; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));

        // Device to Host
        CHECK_CUDA(cudaMemcpy(h_src, d_src, SIZE_BYTES, cudaMemcpyDeviceToHost));

        // Host to Device
        CHECK_CUDA(cudaMemcpy(d_dst, h_src, SIZE_BYTES, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms_per_dtoH_htoD += ms;
    }
    ms_per_dtoH_htoD /= NUM_ITER;

    // Output results
    printf("Average time over %d iterations:\n", NUM_ITER);
    printf("  Device-to-Device (DtoD):   %.3f ms\n", ms_per_dtoD);
    printf("  Device-to-Host + Host-to-Device (DtoH+HtoD): %.3f ms\n",
           ms_per_dtoH_htoD);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    free(h_src);

    return 0;
}
```