```cuda
/*
Aim of the program: 
Measure the bandwidth of the P2P `cudaMemcpy` and compare it to the bandwidth of a manual copy through the host (GPU1 -> Host -> GPU0).

Thinking:
1. The program must run on a system with at least two GPUs. We will query the device count and abort if less than two.
2. Peer-to-peer (P2P) access between the two GPUs is optional; we attempt to enable it on both directions. If it fails, we report that P2P is not supported.
3. Allocate a large buffer (256 MiB) on each GPU using cudaMalloc. Allocate a pinned host buffer (cudaMallocHost) for the host‑mediated copy.
4. Use CUDA events to time the data transfers precisely.
   - For the P2P transfer: time a single cudaMemcpyPeer call that copies data directly from GPU0 to GPU1.
   - For the host‑mediated transfer: time two separate operations: GPU0 → Host and Host → GPU1. The total time is the sum of these two times.
5. Bandwidth is computed as (bytes transferred) / (time in seconds). For the host‑mediated case, we use the size of one transfer (bytes per direction) divided by the sum of times, because the host is the limiting resource.
6. Report both bandwidths in GB/s, along with the transfer times.
7. Clean up all allocated memory and reset the devices.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

int main(void)
{
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount < 2) {
        fprintf(stderr, "This program requires at least 2 GPUs.\n");
        return EXIT_FAILURE;
    }

    const int dev0 = 0;
    const int dev1 = 1;
    const size_t sizeBytes = 256 * 1024 * 1024; // 256 MiB

    // Enable peer access
    bool p2pSupported = true;
    CUDA_CHECK(cudaSetDevice(dev0));
    cudaError_t err = cudaDeviceEnablePeerAccess(dev1, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled || err == cudaSuccess) {
        /* OK */
    } else if (err == cudaErrorPeerAccessNotSupported) {
        p2pSupported = false;
        printf("Peer access from GPU %d to GPU %d not supported.\n", dev0, dev1);
    } else {
        CUDA_CHECK(err);
    }

    CUDA_CHECK(cudaSetDevice(dev1));
    err = cudaDeviceEnablePeerAccess(dev0, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled || err == cudaSuccess) {
        /* OK */
    } else if (err == cudaErrorPeerAccessNotSupported) {
        p2pSupported = false;
        printf("Peer access from GPU %d to GPU %d not supported.\n", dev1, dev0);
    } else {
        CUDA_CHECK(err);
    }

    // Allocate device memory
    void *d0 = NULL, *d1 = NULL;
    CUDA_CHECK(cudaSetDevice(dev0));
    CUDA_CHECK(cudaMalloc(&d0, sizeBytes));
    CUDA_CHECK(cudaMemset(d0, 0xFF, sizeBytes)); // fill with some data

    CUDA_CHECK(cudaSetDevice(dev1));
    CUDA_CHECK(cudaMalloc(&d1, sizeBytes));
    CUDA_CHECK(cudaMemset(d1, 0x00, sizeBytes));

    // Allocate pinned host memory
    void *hBuf = NULL;
    CUDA_CHECK(cudaMallocHost(&hBuf, sizeBytes));

    // Events for timing
    cudaEvent_t startP2P, stopP2P;
    cudaEvent_t startHtoH, stopHtoH; // host transfer to host
    cudaEvent_t startHtoD, stopHtoD; // host to device

    CUDA_CHECK(cudaEventCreate(&startP2P));
    CUDA_CHECK(cudaEventCreate(&stopP2P));

    CUDA_CHECK(cudaEventCreate(&startHtoH));
    CUDA_CHECK(cudaEventCreate(&stopHtoH));

    CUDA_CHECK(cudaEventCreate(&startHtoD));
    CUDA_CHECK(cudaEventCreate(&stopHtoD));

    float p2pTimeMs = 0.0f;
    float hostTimeMs = 0.0f;

    // Measure P2P copy if supported
    if (p2pSupported) {
        CUDA_CHECK(cudaSetDevice(dev0));
        CUDA_CHECK(cudaEventRecord(startP2P, 0));

        CUDA_CHECK(cudaMemcpyPeer(d1, dev1, d0, dev0, sizeBytes));

        CUDA_CHECK(cudaEventRecord(stopP2P, 0));
        CUDA_CHECK(cudaEventSynchronize(stopP2P));

        CUDA_CHECK(cudaEventElapsedTime(&p2pTimeMs, startP2P, stopP2P));
    }

    // Measure host-mediated copy
    // GPU0 -> Host
    CUDA_CHECK(cudaSetDevice(dev0));
    CUDA_CHECK(cudaEventRecord(startHtoH, 0));
    CUDA_CHECK(cudaMemcpy(hBuf, d0, sizeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stopHtoH, 0));
    CUDA_CHECK(cudaEventSynchronize(stopHtoH));
    float htohMs;
    CUDA_CHECK(cudaEventElapsedTime(&htohMs, startHtoH, stopHtoH));

    // Host -> GPU1
    CUDA_CHECK(cudaSetDevice(dev1));
    CUDA_CHECK(cudaEventRecord(startHtoD, 0));
    CUDA_CHECK(cudaMemcpy(d1, hBuf, sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stopHtoD, 0));
    CUDA_CHECK(cudaEventSynchronize(stopHtoD));
    float htodMs;
    CUDA_CHECK(cudaEventElapsedTime(&htodMs, startHtoD, stopHtoD));

    hostTimeMs = htohMs + htodMs;

    // Compute bandwidths
    double p2pBandwidthGBps = 0.0;
    if (p2pSupported && p2pTimeMs > 0.0) {
        p2pBandwidthGBps = (double)sizeBytes / (p2pTimeMs / 1000.0) / (1ULL << 30);
    }

    double hostBandwidthGBps = 0.0;
    if (hostTimeMs > 0.0) {
        hostBandwidthGBps = (double)sizeBytes / (hostTimeMs / 1000.0) / (1ULL << 30);
    }

    printf("Data size: %zu bytes (%.2f MiB)\n", sizeBytes, sizeBytes / (1024.0 * 1024.0));
    if (p2pSupported) {
        printf("P2P transfer time: %.3f ms, Bandwidth: %.2f GB/s\n",
               p2pTimeMs, p2pBandwidthGBps);
    } else {
        printf("P2P transfer not supported.\n");
    }
    printf("Host-mediated transfer time: %.3f ms, Bandwidth: %.2f GB/s\n",
           hostTimeMs, hostBandwidthGBps);

    // Clean up
    CUDA_CHECK(cudaFree(d0));
    CUDA_CHECK(cudaFree(d1));
    CUDA_CHECK(cudaFreeHost(hBuf));

    CUDA_CHECK(cudaEventDestroy(startP2P));
    CUDA_CHECK(cudaEventDestroy(stopP2P));
    CUDA_CHECK(cudaEventDestroy(startHtoH));
    CUDA_CHECK(cudaEventDestroy(stopHtoH));
    CUDA_CHECK(cudaEventDestroy(startHtoD));
    CUDA_CHECK(cudaEventDestroy(stopHtoD));

    CUDA_CHECK(cudaDeviceReset());
    CUDA_CHECK(cudaSetDevice(dev1));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```