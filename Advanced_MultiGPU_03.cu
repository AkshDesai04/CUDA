```cuda
/*
Enable peer-to-peer (P2P) access between two GPUs using `cudaDeviceCanAccessPeer` and `cudaDeviceEnablePeerAccess`. 
Write a program where GPU 0 directly reads from a memory buffer located on GPU 1 using `cudaMemcpyPeerAsync`, bypassing host memory.

Thinking process:
1. Check that at least two GPUs are available.
2. Query P2P capability between device 0 and device 1 using `cudaDeviceCanAccessPeer`.
3. Enable P2P access from device 0 to device 1 via `cudaDeviceEnablePeerAccess`.
4. Allocate a buffer on GPU 1 and initialize it with known data (using a simple kernel).
5. Allocate a buffer on GPU 0 to receive data from GPU 1.
6. Use `cudaMemcpyPeerAsync` to copy data directly from GPU 1 to GPU 0 without going through host memory.
7. Synchronize the stream to ensure the copy completes.
8. Copy the result from GPU 0 to host and verify the contents.
9. Clean up all allocated resources.

The program demonstrates the direct GPU-to-GPU transfer using peer‑to‑peer, which can significantly reduce latency compared to host‑mediated transfers.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Simple kernel to initialize an array on a device with its index value
__global__ void initArray(int *arr, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = idx;
    }
}

int main(void) {
    int devCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&devCount));
    if (devCount < 2) {
        fprintf(stderr, "This program requires at least two GPUs.\n");
        return EXIT_FAILURE;
    }

    const int srcDev = 1; // GPU 1 (source)
    const int dstDev = 0; // GPU 0 (destination)

    // Check P2P capability
    int canAccessPeer = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, dstDev, srcDev));
    if (!canAccessPeer) {
        fprintf(stderr, "GPU %d cannot access peer GPU %d.\n", dstDev, srcDev);
        return EXIT_FAILURE;
    }

    // Enable peer access from dstDev to srcDev
    CHECK_CUDA(cudaSetDevice(dstDev));
    int peerEnabled = 0;
    CHECK_CUDA(cudaDeviceEnablePeerAccess(srcDev, 0));
    peerEnabled = 1;
    if (!peerEnabled) {
        fprintf(stderr, "Failed to enable peer access from GPU %d to GPU %d.\n", dstDev, srcDev);
        return EXIT_FAILURE;
    }

    // Allocate buffer on source GPU (GPU 1)
    size_t N = 1 << 20; // 1M ints
    int *d_src = NULL;
    CHECK_CUDA(cudaSetDevice(srcDev));
    CHECK_CUDA(cudaMalloc((void**)&d_src, N * sizeof(int)));
    // Initialize data on GPU 1
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    initArray<<<blocks, threadsPerBlock>>>(d_src, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Allocate buffer on destination GPU (GPU 0)
    int *d_dst = NULL;
    CHECK_CUDA(cudaSetDevice(dstDev));
    CHECK_CUDA(cudaMalloc((void**)&d_dst, N * sizeof(int)));

    // Create a stream on destination device for the peer copy
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Perform peer copy from GPU 1 to GPU 0
    CHECK_CUDA(cudaMemcpyPeerAsync(d_dst, dstDev, d_src, srcDev,
                                   N * sizeof(int), stream));

    // Synchronize to ensure copy completion
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Copy result back to host for verification
    int *h_res = (int*)malloc(N * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_res, d_dst, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify a few entries
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        if (h_res[i] != i) {
            fprintf(stderr, "Verification failed at index %d: expected %d, got %d\n",
                    i, i, h_res[i]);
            errors++;
        }
    }
    if (errors == 0) {
        printf("Peer copy succeeded. First 10 elements verified correctly.\n");
    } else {
        printf("Peer copy verification failed.\n");
    }

    // Clean up
    free(h_res);
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```