/*
Use Nsight Systems to visualize the dependency. You will see stream2 being idle until the event on stream1 is triggered.

Thinking: The program demonstrates stream synchronization using events in CUDA. Two streams, stream1 and stream2, are created. A simple kernel that writes a value to an array is launched on stream1. After this kernel, an event is recorded on stream1. Then stream2 is made to wait for that event using cudaStreamWaitEvent. A second kernel is launched on stream2, which will only execute after the event is triggered, i.e., after the first kernel completes. This setup will show stream2 idle until the event. The program also uses cudaDeviceSynchronize to ensure all work is done before freeing resources. Timing prints are optional but can help illustrate the idle period. The code is written in CUDA C and can be compiled with nvcc. 
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>

#define CUDA_CHECK(err)                                          \
    do {                                                         \
        cudaError_t err__ = (err);                               \
        if (err__ != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error: %s (%d) at %s:%d\n",    \
                    cudaGetErrorString(err__), err__,           \
                    __FILE__, __LINE__);                        \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

__global__ void kernel1(int *data, int val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

__global__ void kernel2(int *data, int val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

int main() {
    const int N = 1 << 20;          // 1M elements
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    int *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));

    cudaStream_t stream1, stream2;
    cudaEvent_t event;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    CUDA_CHECK(cudaEventCreate(&event));

    // Launch kernel1 on stream1
    auto start1 = std::chrono::high_resolution_clock::now();
    kernel1<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, 1, N);
    CUDA_CHECK(cudaGetLastError());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur1 = end1 - start1;
    std::cout << "Kernel1 enqueued in stream1: " << dur1.count() << " ms" << std::endl;

    // Record event after kernel1 completes on stream1
    CUDA_CHECK(cudaEventRecord(event, stream1));

    // Make stream2 wait for the event
    CUDA_CHECK(cudaStreamWaitEvent(stream2, event, 0));

    // Launch kernel2 on stream2
    auto start2 = std::chrono::high_resolution_clock::now();
    kernel2<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_data, 2, N);
    CUDA_CHECK(cudaGetLastError());
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur2 = end2 - start2;
    std::cout << "Kernel2 enqueued in stream2: " << dur2.count() << " ms" << std::endl;

    // Synchronize to ensure all kernels finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy data back to host and verify
    int *h_data = (int *)malloc(N * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Simple verification
    int ok = 1;
    for (int i = 0; i < N; ++i) {
        if (h_data[i] != 2) { ok = 0; break; }
    }
    std::cout << "Verification " << (ok ? "PASSED" : "FAILED") << std::endl;

    // Cleanup
    free(h_data);
    CUDA_CHECK(cudaEventDestroy(event));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
