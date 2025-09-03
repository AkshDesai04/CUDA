```cpp
/*
Aim: Extend the halo exchange pattern to a ring of all available GPUs, where each GPU communicates with two neighbors.

Thinking:
The goal is to perform a 1‑D halo exchange for a data domain that has been decomposed across all GPUs available on the system.  
Each GPU holds a 2‑D sub‑array of size (N+2)×(N+2) where the outermost rows (row 0 and row N+1) are the halo regions that must be kept in sync with the neighbouring GPUs in a ring topology.  
For a ring, GPU i communicates with GPU left = (i-1+P)%P on its left side and with GPU right = (i+1)%P on its right side.  The halo exchange therefore consists of two separate peer‑to‑peer copies per GPU:
1. Copy the first interior row (row 1) of GPU i to the bottom halo row (row N+1) of GPU left, and vice‑versa the bottom interior row of GPU left to the top halo row of GPU i.
2. Copy the last interior row (row N) of GPU i to the top halo row (row 0) of GPU right, and vice‑versa the top interior row of GPU right to the bottom halo row of GPU i.

To achieve this efficiently we use CUDA Peer‑to‑Peer (P2P) memory copies with `cudaMemcpyPeerAsync`.  
We allocate a separate stream for the two directions (top‑bottom and bottom‑top) so that the copies can overlap.  Peer access must be enabled between each pair of GPUs before performing the copy.  After the halo exchange we simply synchronize the streams and then clean up.  No compute kernel is required for the exchange itself, but a placeholder can be added if needed.

The implementation below:
- Detects the number of GPUs (`P`) on the system.
- Allocates a double array of size (N+2)×(N+2) on each GPU.
- Enables P2P access between all pairs of GPUs.
- Performs the halo exchange using two asynchronous copies per GPU.
- Waits for all copies to finish and then releases resources.

This pattern can be extended to higher‑dimensional decompositions or to actual compute kernels that use the updated halo data.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024  // interior domain size
#define WIDTH (N+2) // including halos

// Helper macro for checking CUDA errors
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        fprintf(stderr, "Need at least two GPUs for ring halo exchange.\n");
        return EXIT_FAILURE;
    }

    printf("Detected %d GPUs.\n", deviceCount);

    // Allocate data on each GPU
    double *d_arrays[deviceCount];
    for (int dev = 0; dev < deviceCount; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        size_t bytes = sizeof(double) * WIDTH * WIDTH;
        CUDA_CHECK(cudaMalloc(&d_arrays[dev], bytes));
        // Optional: initialize data (omitted for brevity)
    }

    // Enable P2P access between all pairs of GPUs
    for (int src = 0; src < deviceCount; ++src) {
        CUDA_CHECK(cudaSetDevice(src));
        for (int dst = 0; dst < deviceCount; ++dst) {
            if (src == dst) continue;
            int access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&access, src, dst));
            if (access) {
                CUDA_CHECK(cudaDeviceEnablePeerAccess(dst, 0));
            } else {
                fprintf(stderr, "GPUs %d and %d cannot access each other.\n", src, dst);
            }
        }
    }

    // Create streams for each device
    cudaStream_t stream_top[deviceCount];
    cudaStream_t stream_bottom[deviceCount];
    for (int dev = 0; dev < deviceCount; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaStreamCreate(&stream_top[dev]));
        CUDA_CHECK(cudaStreamCreate(&stream_bottom[dev]));
    }

    // Perform halo exchange
    for (int dev = 0; dev < deviceCount; ++dev) {
        int left  = (dev - 1 + deviceCount) % deviceCount;
        int right = (dev + 1) % deviceCount;

        size_t row_bytes = sizeof(double) * WIDTH;

        // 1. Send first interior row (row 1) to left's bottom halo (row N+1)
        //    and receive left's last interior row (row N) into own top halo (row 0)
        //    (left -> dev: row N -> row 0)
        CUDA_CHECK(cudaMemcpyPeerAsync(
            d_arrays[dev] + 0 * WIDTH,            // destination: row 0 of dev
            dev,
            d_arrays[left] + (N) * WIDTH,         // source: row N of left
            left,
            row_bytes,
            stream_top[dev]
        ));

        CUDA_CHECK(cudaMemcpyPeerAsync(
            d_arrays[left] + (N+1) * WIDTH,       // destination: row N+1 of left
            left,
            d_arrays[dev] + (1) * WIDTH,          // source: row 1 of dev
            dev,
            row_bytes,
            stream_top[dev]
        ));

        // 2. Send last interior row (row N) to right's top halo (row 0)
        //    and receive right's first interior row (row 1) into own bottom halo (row N+1)
        CUDA_CHECK(cudaMemcpyPeerAsync(
            d_arrays[dev] + (N+1) * WIDTH,        // destination: row N+1 of dev
            dev,
            d_arrays[right] + (1) * WIDTH,        // source: row 1 of right
            right,
            row_bytes,
            stream_bottom[dev]
        ));

        CUDA_CHECK(cudaMemcpyPeerAsync(
            d_arrays[right] + 0 * WIDTH,          // destination: row 0 of right
            right,
            d_arrays[dev] + (N) * WIDTH,          // source: row N of dev
            dev,
            row_bytes,
            stream_bottom[dev]
        ));
    }

    // Wait for all streams to finish
    for (int dev = 0; dev < deviceCount; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaStreamSynchronize(stream_top[dev]));
        CUDA_CHECK(cudaStreamSynchronize(stream_bottom[dev]));
    }

    // Optional: Verify data (omitted)

    // Clean up
    for (int dev = 0; dev < deviceCount; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaFree(d_arrays[dev]));
        CUDA_CHECK(cudaStreamDestroy(stream_top[dev]));
        CUDA_CHECK(cudaStreamDestroy(stream_bottom[dev]));
    }

    printf("Halo exchange completed successfully.\n");
    return EXIT_SUCCESS;
}
```