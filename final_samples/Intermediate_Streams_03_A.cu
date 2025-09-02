/*
Aim: Calculate the speedup factor (Sequential Time / Streamed Time).

Thinking:
The goal is to compare the performance of a simple element‑wise vector addition implemented
sequentially on the CPU against an implementation that uses CUDA streams on the GPU.
The program will:
1. Generate two large input arrays (A and B) filled with random floating point numbers.
2. Perform the addition on the CPU while measuring the elapsed time with std::chrono.
3. Allocate device memory for A, B, and C.
4. Create two CUDA streams to overlap data transfers:
   - Stream 0 copies A from host to device.
   - Stream 1 copies B from host to device.
5. After both copies finish, launch the vector addition kernel in the default stream.
   The kernel will compute C = A + B.
6. Copy the result back to host asynchronously on the default stream.
7. Measure the total GPU elapsed time (including both transfer and kernel execution)
   using CUDA events.
8. Compute the speedup factor as (Sequential CPU time) / (GPU streamed time)
   and print it to the console.
The program is self‑contained, compiles to a .cu file, and contains all necessary includes,
error checking, and cleanup. It demonstrates basic stream usage and performance measurement
in CUDA C.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <chrono>

// Error checking macro
#define CHECK_CUDA(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

// Vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main()
{
    const int N = 1 << 24; // 16 million elements
    const size_t bytes = N * sizeof(float);

    // Allocate pinned host memory for faster async transfers
    float *h_A, *h_B, *h_C;
    CHECK_CUDA(cudaMallocHost(&h_A, bytes));
    CHECK_CUDA(cudaMallocHost(&h_B, bytes));
    CHECK_CUDA(cudaMallocHost(&h_C, bytes));

    // Initialize input arrays with random data
    srand(0);
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // ==================== Sequential CPU implementation ====================
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        h_C[i] = h_A[i] + h_B[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_elapsed = cpu_end - cpu_start;
    double cpu_time_sec = cpu_elapsed.count();

    // ==================== GPU implementation with streams ====================
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Create streams
    cudaStream_t stream0, stream1;
    CHECK_CUDA(cudaStreamCreate(&stream0));
    CHECK_CUDA(cudaStreamCreate(&stream1));

    // Events for timing
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    // Record start event
    CHECK_CUDA(cudaEventRecord(startEvent, 0));

    // Asynchronously copy A to device on stream0
    CHECK_CUDA(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream0));

    // Asynchronously copy B to device on stream1
    CHECK_CUDA(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, stream1));

    // Ensure both copies are finished before kernel launch
    CHECK_CUDA(cudaStreamSynchronize(stream0));
    CHECK_CUDA(cudaStreamSynchronize(stream1));

    // Launch kernel on default stream (synchronizes with previous streams)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    // Asynchronously copy result back to host on default stream
    CHECK_CUDA(cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, 0));

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    float gpu_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_time_ms, startEvent, stopEvent));
    double gpu_time_sec = gpu_time_ms / 1000.0;

    // Compute speedup factor
    double speedup = cpu_time_sec / gpu_time_sec;

    // Output results
    printf("Sequential CPU time: %.6f seconds\n", cpu_time_sec);
    printf("GPU Streamed time: %.6f seconds\n", gpu_time_sec);
    printf("Speedup factor (CPU/GPU): %.2f\n", speedup);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaStreamDestroy(stream0));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_C));

    return 0;
}
