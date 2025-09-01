/*
 * Aim: Perform a round trip: HtoD, then DtoH. Time the entire process.
 * 
 * The goal of this program is to measure the total time taken for a data
 * transfer round-trip between the host (CPU) and the device (GPU) using
 * CUDA.  The program:
 * 1. Allocates a large array on the host and initializes it.
 * 2. Allocates a corresponding array on the device.
 * 3. Starts a CUDA event timer.
 * 4. Copies the data from host to device (HtoD).
 * 5. Copies the data back from device to host (DtoH).
 * 6. Stops the timer and reports the elapsed time in milliseconds.
 *
 * This simple round-trip benchmark helps evaluate the bandwidth
 * and latency characteristics of the PCIe or NVLink interconnect.
 *
 * Thinking process:
 * - Use cudaEvent_t for precise GPU timing, as it measures time spent in the
 *   device copy operations only, ignoring CPU overhead.
 * - Ensure all CUDA calls are checked for errors using a helper macro.
 * - Allocate enough data to give a measurable time, e.g. 100 MB.
 * - Keep the code self-contained with a single main function.
 * - Include necessary headers and clean up resources at the end.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Error checking macro */
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void)
{
    const size_t N = 100 * 1024 * 1024; // 100 million elements (~400 MB)
    size_t bytes = N * sizeof(float);

    /* Allocate and initialize host memory */
    float *h_data = (float*)malloc(bytes);
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    /* Allocate device memory */
    float *d_data = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_data, bytes));

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* Record start event */
    CHECK_CUDA(cudaEventRecord(start, 0));

    /* Host to Device copy */
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    /* Device to Host copy */
    CHECK_CUDA(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    /* Record stop event and synchronize */
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    /* Compute elapsed time */
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("Round-trip (HtoD + DtoH) time for %zu floats (%zu bytes): %.3f ms\n",
           N, bytes, ms);

    /* Clean up */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);

    return EXIT_SUCCESS;
}
