/*
Aim of the program: Compare your measured bandwidth to the theoretical peak bandwidth of your GPU's PCIe bus.

Thinking:
- We need to write a CUDA C program that measures the bandwidth of PCIe transfers between host and device.
- The typical approach is to perform a large cudaMemcpy from host to device (and optionally back), timing the operation with cudaEvent_t.
- We'll allocate pinned host memory (cudaMallocHost) to avoid page fault overhead, and device memory with cudaMalloc.
- For repeatability, we perform multiple copies and average the time.
- Bandwidth is computed as (bytes transferred / time_in_seconds).
- The theoretical peak bandwidth depends on PCIe generation and lanes. For a typical modern GPU using PCIe 3.0 x16, the theoretical max is about 15.75 GB/s (per direction). For PCIe 4.0 x16, it's roughly 31.5 GB/s. We'll assume PCIe 3.0 x16 unless the user changes the constant.
- The program will print:
  * Size of the transfer.
  * Measured time for host-to-device and device-to-host.
  * Measured bandwidth for each direction.
  * Theoretical peak bandwidth.
  * A simple comparison: whether measured < theoretical (expected) and the percentage achieved.
- We'll also include basic error checking for CUDA API calls using a macro.
- The code is self-contained and can be compiled with nvcc.
- The program will not use any external libraries beyond the CUDA runtime.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    #call, __FILE__, __LINE__,                    \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                          \
    } while (0)

int main(void) {
    /* Transfer size: 256 MB (change if needed) */
    const size_t TRANSFER_SIZE = 256 * 1024 * 1024; // 256 MB
    const int NUM_RUNS = 10; // number of timed copies

    /* Allocate pinned host memory */
    void *h_buf = NULL;
    CUDA_CHECK(cudaMallocHost(&h_buf, TRANSFER_SIZE));

    /* Allocate device memory */
    void *d_buf = NULL;
    CUDA_CHECK(cudaMalloc(&d_buf, TRANSFER_SIZE));

    /* Initialize host buffer with dummy data */
    for (size_t i = 0; i < TRANSFER_SIZE; ++i) {
        ((char*)h_buf)[i] = (char)(i % 256);
    }

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Warm-up copy to hide cold start overhead */
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, TRANSFER_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, TRANSFER_SIZE, cudaMemcpyDeviceToHost));

    /* Measure Host -> Device copy */
    float elapsed_htd = 0.0f;
    for (int i = 0; i < NUM_RUNS; ++i) {
        CUDA_CHECK(cudaEventRecord(start, 0));
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, TRANSFER_SIZE, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        elapsed_htd += ms;
    }
    elapsed_htd /= NUM_RUNS; // average in ms

    /* Measure Device -> Host copy */
    float elapsed_dth = 0.0f;
    for (int i = 0; i < NUM_RUNS; ++i) {
        CUDA_CHECK(cudaEventRecord(start, 0));
        CUDA_CHECK(cudaMemcpy(h_buf, d_buf, TRANSFER_SIZE, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        elapsed_dth += ms;
    }
    elapsed_dth /= NUM_RUNS; // average in ms

    /* Compute bandwidths in GB/s */
    double time_sec_htd = elapsed_htd / 1000.0;
    double time_sec_dth = elapsed_dth / 1000.0;
    double bw_htd = (double)TRANSFER_SIZE / time_sec_htd / (1 << 30); // GB/s
    double bw_dth = (double)TRANSFER_SIZE / time_sec_dth / (1 << 30); // GB/s

    /* Theoretical peak for PCIe 3.0 x16 is approximately 15.75 GB/s per direction */
    const double theoretical_peak = 15.75; // GB/s

    /* Output results */
    printf("PCIe Bandwidth Measurement (512 MB total transfer size per direction):\n");
    printf("  Host->Device average time: %.3f ms, Bandwidth: %.3f GB/s\n", elapsed_htd, bw_htd);
    printf("  Device->Host average time: %.3f ms, Bandwidth: %.3f GB/s\n", elapsed_dth, bw_dth);
    printf("Theoretical peak PCIe 3.0 x16 bandwidth: %.2f GB/s per direction\n", theoretical_peak);
    printf("Host->Device achieved: %.2f%% of theoretical peak\n", bw_htd / theoretical_peak * 100.0);
    printf("Device->Host achieved: %.2f%% of theoretical peak\n", bw_dth / theoretical_peak * 100.0);

    /* Clean up */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFreeHost(h_buf));

    return 0;
}
