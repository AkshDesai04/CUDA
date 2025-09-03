/*
Aim: Use Nsight Systems to visualize the dependency. You will see stream2 being idle until the event on stream1 is triggered.

Thought Process:
- The goal is to demonstrate a clear dependency between two CUDA streams using events.
- We'll create two streams: `stream1` and `stream2`.
- A simple kernel (`slowKernel`) will run on `stream1`. It will perform a long loop to simulate a time-consuming operation.
- After `slowKernel` completes, we record an event `event1` on `stream1`.
- `stream2` will launch a second kernel (`fastKernel`) that depends on the completion of `event1`. We enforce this dependency with `cudaStreamWaitEvent`.
- As a result, when visualizing with Nsight Systems, `stream2` will appear idle until `event1` is triggered by the completion of `slowKernel` on `stream1`.
- We'll include minimal error checking and device memory allocations to keep the example focused on the stream/event dependency.
- Nsight Systems can be launched with `nsys profile ./dependency_example` to observe the timeline.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Simple kernel that performs a time-consuming operation
__global__ void slowKernel(int *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simulate workload
        for (int i = 0; i < 1 << 20; ++i) {  // ~1 million iterations
            data[idx] += 1;
        }
    }
}

// Simple kernel that performs a quick operation
__global__ void fastKernel(int *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 2;
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(int);

    int *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemset(d_data, 0, bytes));

    // Create two streams
    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    // Launch slowKernel on stream1
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    slowKernel<<<gridSize, blockSize, 0, stream1>>>(d_data, N);
    CHECK_CUDA(cudaGetLastError());

    // Record an event after slowKernel completes
    cudaEvent_t event1;
    CHECK_CUDA(cudaEventCreate(&event1));
    CHECK_CUDA(cudaEventRecord(event1, stream1));

    // Ensure stream2 waits for event1 before launching fastKernel
    CHECK_CUDA(cudaStreamWaitEvent(stream2, event1, 0));

    // Launch fastKernel on stream2
    fastKernel<<<gridSize, blockSize, 0, stream2>>>(d_data, N);
    CHECK_CUDA(cudaGetLastError());

    // Wait for both streams to finish
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));

    // Clean up
    CHECK_CUDA(cudaEventDestroy(event1));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaFree(d_data));

    printf("Execution complete. Check Nsight Systems timeline for dependency visualization.\n");
    return 0;
}
