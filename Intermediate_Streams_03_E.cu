```cuda
/*
Make sure you are using pinned host memory. The overlap will not occur with pageable memory.

This program demonstrates the use of pinned (page-locked) host memory in CUDA to achieve
asynchronous memory transfers that can overlap with kernel execution. The key idea is to
allocate host buffers with cudaMallocHost(), which guarantees that the memory is pinned
and thus can be transferred asynchronously without being paged in or out by the operating
system. By using separate CUDA streams and event synchronization we show that:

1. The device-to-host copy (async copy back) can overlap with kernel execution
   because the host memory is pinned and the transfer can be performed concurrently
   with the GPU compute.  
2. If we used regular (pageable) host memory instead, the asynchronous copies
   would be serialized, and no overlap would be observed.

The program performs a simple element‑wise vector addition:
    C = A + B
on a large array. It measures the time spent in copying data to the device,
executing the kernel, and copying results back to the host.  The timings are
reported for both the naive sequential case (single stream) and the overlapped
case (separate copy and compute streams).  The overlapped case should
demonstrate reduced total time when pinned memory is used.

Author: OpenAI ChatGPT
Date: 2025‑09‑04
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 24)          // Number of elements (~16 million)
#define BLOCK_SIZE 256

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

// Error checking macro
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main()
{
    size_t bytes = N * sizeof(float);

    // ---------- Allocate pinned host memory ----------
    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost((void**)&h_A, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_B, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_C, bytes));

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // ---------- Allocate device memory ----------
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // ---------- Create streams ----------
    cudaStream_t copyStream, computeStream;
    CUDA_CHECK(cudaStreamCreate(&copyStream));
    CUDA_CHECK(cudaStreamCreate(&computeStream));

    // ---------- Create events for synchronization ----------
    cudaEvent_t copyEvent, computeEvent;
    CUDA_CHECK(cudaEventCreate(&copyEvent));
    CUDA_CHECK(cudaEventCreate(&computeEvent));

    // ---------- Timing events ----------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ---------- Overlapped execution ----------
    CUDA_CHECK(cudaEventRecord(start, 0));  // Record start on default stream

    // 1. Asynchronously copy A and B to device in copyStream
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, copyStream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, copyStream));

    // 2. Record event after copy in copyStream
    CUDA_CHECK(cudaEventRecord(copyEvent, copyStream));

    // 3. Make computeStream wait for copyEvent
    CUDA_CHECK(cudaStreamWaitEvent(computeStream, copyEvent, 0));

    // 4. Launch kernel on computeStream
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorAdd<<<gridSize, BLOCK_SIZE, 0, computeStream>>>(d_A, d_B, d_C, N);

    // 5. Record event after kernel in computeStream
    CUDA_CHECK(cudaEventRecord(computeEvent, computeStream));

    // 6. Make copyStream wait for kernel completion before copying back
    CUDA_CHECK(cudaStreamWaitEvent(copyStream, computeEvent, 0));

    // 7. Asynchronously copy result back to host in copyStream
    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, copyStream));

    // 8. Record stop event on default stream after copy back is finished
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for all operations to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time
    float msTotal = 0;
    CUDA_CHECK(cudaEventElapsedTime(&msTotal, start, stop));
    printf("Total elapsed time with overlap (pinned memory): %.3f ms\n", msTotal);

    // ---------- Validation ----------
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            printf("Mismatch at index %d: got %f, expected %f\n", i, h_C[i], expected);
            correct = false;
            break;
        }
    }
    printf("Result validation: %s\n", correct ? "PASS" : "FAIL");

    // ---------- Cleanup ----------
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    CUDA_CHECK(cudaStreamDestroy(copyStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(copyEvent));
    CUDA_CHECK(cudaEventDestroy(computeEvent));

    return 0;
}
```