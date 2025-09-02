/*
Aim: Is pinned memory required for `cudaMemcpyAsync` to be truly asynchronous and overlap with kernel execution? (Yes). Explain why.

Thinking:
To allow a transfer initiated by cudaMemcpyAsync to overlap with kernel execution, the transfer must be able to proceed in the background without the host thread waiting for it to finish. CUDA achieves this by using DMA engines that can directly read from or write to host memory. For the DMA engine to read/write from host memory efficiently, that memory must be page‑locked (pinned). Pinned memory guarantees that the memory resides in a contiguous physical region and cannot be paged out to disk, so the GPU can access it directly and the CUDA runtime can issue the transfer on a stream without staging data through intermediate pinned buffers.

If the memory is not pinned, the CUDA runtime has to perform a staging copy: it allocates an internal pinned buffer, copies the data from the user buffer to that buffer (synchronously on the host), then issues the device transfer. The initial synchronous copy negates the overlap benefit, and the transfer itself may be queued but cannot start until the host copy completes. Therefore, true asynchrony and overlap with kernel execution require that both the source and destination pointers used in cudaMemcpyAsync refer to pinned host memory. In addition, the stream used for the copy must be the same stream used for the kernel launch (or the streams must be synchronized appropriately) for the GPU to execute them concurrently. The following code demonstrates this by comparing overlapping behavior with pinned versus non‑pinned memory.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 20)   // 1M elements
#define BLOCK_SIZE 256

__global__ void addKernel(int *data, int val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += val;
    }
}

void checkCudaErr(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int *h_src, *h_dst;
    int *d_dst;
    size_t size = N * sizeof(int);

    // Allocate non‑pinned host memory
    h_src = (int *)malloc(size);
    h_dst = (int *)malloc(size);
    for (int i = 0; i < N; ++i) h_src[i] = i;

    // Allocate pinned host memory
    int *h_pinned_src, *h_pinned_dst;
    cudaMallocHost((void **)&h_pinned_src, size);
    cudaMallocHost((void **)&h_pinned_dst, size);
    for (int i = 0; i < N; ++i) h_pinned_src[i] = i;

    // Allocate device memory
    cudaMalloc((void **)&d_dst, size);
    checkCudaErr("cudaMalloc d_dst");

    // Create streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // ---------- Test with non‑pinned memory ----------
    printf("Testing with non‑pinned memory (should NOT overlap):\n");

    cudaEventRecord(startEvent, stream1);
    cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream1);
    addKernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream1>>>(d_dst, 1);
    cudaMemcpyAsync(h_dst, d_dst, size, cudaMemcpyDeviceToHost, stream1);
    cudaEventRecord(stopEvent, stream1);

    cudaEventSynchronize(stopEvent);
    float ms;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("Elapsed time: %f ms (no overlap expected)\n", ms);

    // Verify result
    cudaMemcpy(h_dst, d_dst, size, cudaMemcpyDeviceToHost);
    int correct = 1;
    for (int i = 0; i < N; ++i) {
        if (h_dst[i] != i + 1) {
            correct = 0;
            break;
        }
    }
    printf("Result correct: %s\n\n", correct ? "YES" : "NO");

    // ---------- Test with pinned memory ----------
    printf("Testing with pinned memory (should overlap):\n");

    cudaEventRecord(startEvent, stream2);
    cudaMemcpyAsync(d_dst, h_pinned_src, size, cudaMemcpyHostToDevice, stream2);
    addKernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream2>>>(d_dst, 1);
    cudaMemcpyAsync(h_pinned_dst, d_dst, size, cudaMemcpyDeviceToHost, stream2);
    cudaEventRecord(stopEvent, stream2);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("Elapsed time: %f ms (overlap expected)\n", ms);

    // Verify result
    int correct_pinned = 1;
    for (int i = 0; i < N; ++i) {
        if (h_pinned_dst[i] != i + 1) {
            correct_pinned = 0;
            break;
        }
    }
    printf("Result correct: %s\n", correct_pinned ? "YES" : "NO");

    // Cleanup
    free(h_src);
    free(h_dst);
    cudaFreeHost(h_pinned_src);
    cudaFreeHost(h_pinned_dst);
    cudaFree(d_dst);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}
