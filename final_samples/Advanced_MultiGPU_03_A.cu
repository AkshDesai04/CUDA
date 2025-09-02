```c
/*
 * Aim of the program:
 * Measure the bandwidth of the P2P `cudaMemcpy` and compare it to the bandwidth
 * of a manual copy through the host (GPU1 -> Host -> GPU0).
 *
 * Thinking and design:
 * 1. Determine the number of available CUDA devices. The benchmark requires at
 *    least two GPUs. If fewer than two are available, the program exits with a
 *    message.
 *
 * 2. Allocate a large buffer on each GPU. We use 512 MB (1<<29 bytes) which
 *    provides enough data to get a stable measurement. Device 0 holds the
 *    destination buffer; Device 1 holds the source buffer.
 *
 * 3. Enable peer-to-peer (P2P) access between the two GPUs using
 *    `cudaDeviceEnablePeerAccess`. P2P may not be available on all hardware,
 *    so we check the return code. If P2P is not supported, we report it and
 *    skip the P2P copy measurement, but still perform the host path copy.
 *
 * 4. Initialize the source buffer on GPU1. We simply set all bytes to 0xAA
 *    using `cudaMemset`. This is fast and sufficient for bandwidth measurement.
 *
 * 5. Timing is performed with CUDA events (`cudaEvent_t`). Each copy operation
 *    is wrapped by a start/stop event pair, and the elapsed time is obtained
 *    via `cudaEventElapsedTime` (returns milliseconds).
 *
 * 6. P2P copy measurement:
 *    - Set the current device to GPU1.
 *    - Record start event.
 *    - Perform `cudaMemcpyPeer` from GPU1 buffer to GPU0 buffer.
 *    - Record stop event.
 *    - Compute elapsed time and bandwidth (bytes / seconds).
 *
 * 7. Host copy measurement:
 *    - Allocate a pinned host buffer of the same size using `cudaHostAlloc`
 *      to ensure high throughput.
 *    - Copy from GPU1 to host (`cudaMemcpy`).
 *    - Copy from host to GPU0 (`cudaMemcpy`).
 *    - The two transfers are timed separately or together; here we time
 *      the entire round‑trip in a single measurement.
 *
 * 8. Print the measured bandwidths in GB/s for both P2P and host‑mediated
 *    transfers. Also report whether P2P was available.
 *
 * 9. Clean up: free all device and host memory, disable peer access,
 *    and destroy CUDA events.
 *
 * 10. Error handling: All CUDA API calls are wrapped in a macro that checks
 *     the returned `cudaError_t` and prints an informative message before
 *     exiting. This ensures that any failure is clearly reported.
 *
 * The program is self‑contained and can be compiled with:
 *     nvcc -arch=sm_70 -o bandwidth_test bandwidth_test.cu
 * (Adjust the architecture as needed for your GPUs.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    const size_t bufSize = 512ULL * 1024 * 1024;  // 512 MB
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < 2) {
        fprintf(stderr, "Need at least 2 CUDA devices, found %d.\n", deviceCount);
        return EXIT_FAILURE;
    }

    int dev0 = 0;
    int dev1 = 1;

    /* Allocate device memory */
    void *devBuf0 = NULL, *devBuf1 = NULL;
    CUDA_CHECK(cudaSetDevice(dev0));
    CUDA_CHECK(cudaMalloc(&devBuf0, bufSize));
    CUDA_CHECK(cudaSetDevice(dev1));
    CUDA_CHECK(cudaMalloc(&devBuf1, bufSize));

    /* Enable peer access */
    int canAccessPeer = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, dev0, dev1));
    int p2pAvailable = 0;
    if (canAccessPeer) {
        /* Enable peer access from dev1 to dev0 */
        CUDA_CHECK(cudaSetDevice(dev1));
        cudaError_t err = cudaDeviceEnablePeerAccess(dev0, 0);
        if (err == cudaSuccess) {
            p2pAvailable = 1;
        } else if (err == cudaErrorPeerAccessAlreadyEnabled) {
            p2pAvailable = 1;
        } else {
            fprintf(stderr, "Failed to enable P2P from device %d to %d: %s\n",
                    dev1, dev0, cudaGetErrorString(err));
        }
    }

    /* Initialize source buffer on GPU1 */
    CUDA_CHECK(cudaSetDevice(dev1));
    CUDA_CHECK(cudaMemset(devBuf1, 0xAA, bufSize));

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Measure P2P copy if available */
    float p2pTimeMs = 0.0f;
    if (p2pAvailable) {
        CUDA_CHECK(cudaSetDevice(dev1));  // Set to source device
        CUDA_CHECK(cudaEventRecord(start, 0));
        CUDA_CHECK(cudaMemcpyPeer(devBuf0, dev0, devBuf1, dev1, bufSize));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&p2pTimeMs, start, stop));
    }

    /* Measure host-mediated copy */
    /* Allocate pinned host memory */
    void *hostBuf = NULL;
    CUDA_CHECK(cudaHostAlloc(&hostBuf, bufSize, cudaHostAllocDefault));

    CUDA_CHECK(cudaEventRecord(start, 0));
    /* Copy from GPU1 to host */
    CUDA_CHECK(cudaMemcpy(hostBuf, devBuf1, bufSize, cudaMemcpyDeviceToHost));
    /* Copy from host to GPU0 */
    CUDA_CHECK(cudaMemcpy(devBuf0, hostBuf, bufSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float hostTimeMs;
    CUDA_CHECK(cudaEventElapsedTime(&hostTimeMs, start, stop));

    /* Compute bandwidths */
    double p2pBandwidthGBs = 0.0;
    if (p2pAvailable) {
        p2pBandwidthGBs = (double)bufSize / (p2pTimeMs / 1e3) / 1e9;
    }
    double hostBandwidthGBs = (double)bufSize / (hostTimeMs / 1e3) / 1e9;

    /* Print results */
    printf("Buffer size: %.2f MB\n", (double)bufSize / (1024 * 1024));
    if (p2pAvailable) {
        printf("P2P copy time: %.3f ms, bandwidth: %.3f GB/s\n",
               p2pTimeMs, p2pBandwidthGBs);
    } else {
        printf("P2P copy not available.\n");
    }
    printf("Host-mediated copy time: %.3f ms, bandwidth: %.3f GB/s\n",
           hostTimeMs, hostBandwidthGBs);

    /* Cleanup */
    CUDA_CHECK(cudaFree(devBuf0));
    CUDA_CHECK(cudaFree(devBuf1));
    CUDA_CHECK(cudaFreeHost(hostBuf));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    /* Disable peer access if enabled */
    if (p2pAvailable) {
        CUDA_CHECK(cudaSetDevice(dev1));
        CUDA_CHECK(cudaDeviceDisablePeerAccess(dev0));
    }

    return EXIT_SUCCESS;
}
```