```cpp
/*
Aim of the program:
Measure the performance improvement of the streamed version over the sequential version. The overlap of copy and execution should result in a significant speedup, hiding the data transfer latency.

Thinking and design decisions:
1. The program will perform a simple vector addition on large arrays to expose both memory transfer and compute costs.
2. Two implementations are compared:
   - Sequential: single large cudaMemcpy (H2D), kernel launch, cudaMemcpy (D2H) in order.
   - Streamed: the data is divided into several chunks, each processed in its own CUDA stream. For each chunk the host-to-device copy, kernel launch, and device-to-host copy are performed asynchronously within that stream. Because the streams are independent, the GPU can overlap the memory transfers of one chunk with the kernel execution of another, potentially hiding transfer latency.
3. Pinned host memory (cudaHostAlloc) is used to allow true asynchronous memory copies.
4. The host measures total elapsed time for each method using cuda events.
5. The vector addition kernel is trivial: C[i] = A[i] + B[i].
6. The program reports the times and computes the speedup factor.
7. Error checking is performed after each CUDA API call and kernel launch.
8. The code is written in CUDA C, ready to compile with nvcc and produce a .cu file.
*/

#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"\n"; \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

__global__ void vectorAdd(const float* A, const float* B, float* C, size_t offset, size_t len)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < offset + len) {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const size_t N = 1 << 28;            // Number of elements (~268 million)
    const size_t NUM_STREAMS = 4;        // Number of CUDA streams for pipelining
    const size_t bytes = N * sizeof(float);

    std::cout << "Vector size: " << N << " elements (" << bytes / (1 << 20) << " MB)\n";
    std::cout << "Using " << NUM_STREAMS << " streams for pipelined version.\n\n";

    // Allocate pinned host memory
    float *h_A, *h_B, *h_C;
    CHECK_CUDA(cudaHostAlloc((void**)&h_A, bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&h_B, bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&h_C, bytes, cudaHostAllocDefault));

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // ==================== Sequential Version ====================
    cudaEvent_t seq_start, seq_stop;
    CHECK_CUDA(cudaEventCreate(&seq_start));
    CHECK_CUDA(cudaEventCreate(&seq_stop));

    CHECK_CUDA(cudaEventRecord(seq_start, 0));

    // Host to Device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Kernel launch
    const size_t threadsPerBlock = 256;
    const size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, 0, N);
    CHECK_CUDA(cudaGetLastError());

    // Device to Host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(seq_stop, 0));
    CHECK_CUDA(cudaEventSynchronize(seq_stop));

    float seq_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&seq_time_ms, seq_start, seq_stop));

    std::cout << "Sequential version time: " << seq_time_ms << " ms\n";

    // ==================== Streamed Version ====================
    cudaEvent_t stream_start, stream_stop;
    CHECK_CUDA(cudaEventCreate(&stream_start));
    CHECK_CUDA(cudaEventCreate(&stream_stop));

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Determine chunk sizes
    size_t chunk_size = N / NUM_STREAMS;
    size_t remainder = N % NUM_STREAMS;

    CHECK_CUDA(cudaEventRecord(stream_start, 0));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        size_t offset = i * chunk_size;
        size_t len = (i == NUM_STREAMS - 1) ? (chunk_size + remainder) : chunk_size;

        // Asynchronous Host to Device copy
        CHECK_CUDA(cudaMemcpyAsync(d_A + offset, h_A + offset, len * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_B + offset, h_B + offset, len * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[i]));

        // Kernel launch in same stream
        const size_t chunk_blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<chunk_blocks, threadsPerBlock, 0, streams[i]>>>(d_A + offset, d_B + offset,
                                                                   d_C + offset, 0, len);
        CHECK_CUDA(cudaGetLastError());

        // Asynchronous Device to Host copy
        CHECK_CUDA(cudaMemcpyAsync(h_C + offset, d_C + offset, len * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[i]));
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaEventRecord(stream_stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stream_stop));

    float stream_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&stream_time_ms, stream_start, stream_stop));

    std::cout << "Streamed version time:   " << stream_time_ms << " ms\n";

    // Speedup calculation
    float speedup = seq_time_ms / stream_time_ms;
    std::cout << "Speedup (sequential / streamed): " << speedup << "x\n";

    // Clean up
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaEventDestroy(seq_start));
    CHECK_CUDA(cudaEventDestroy(seq_stop));
    CHECK_CUDA(cudaEventDestroy(stream_start));
    CHECK_CUDA(cudaEventDestroy(stream_stop));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_C));

    return 0;
}
```