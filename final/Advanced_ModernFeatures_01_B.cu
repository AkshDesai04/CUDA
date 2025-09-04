```cpp
/*
Aim of the program:
Use `cudaMemPrefetchAsync` to proactively migrate data to a specific processor (CPU or a specific GPU) before it's accessed to hide migration latency.

Thinking:
- To demonstrate `cudaMemPrefetchAsync`, we use Unified Memory (`cudaMallocManaged`) because it allows the memory to be visible to both CPU and GPU.
- The program allocates a large array of floats and initializes it on the CPU.
- Before launching a simple kernel that adds two arrays, we proactively prefetch the managed memory to the GPU device using `cudaMemPrefetchAsync`. This moves the data to GPU memory ahead of the kernel launch, hiding the migration latency that would otherwise occur at first GPU access.
- After the kernel finishes, we prefetch the data back to the CPU in the same proactive way, so that the CPU can read the results without waiting for the implicit migration.
- CUDA events are used to time the kernel execution and the migration prefetches. By comparing the timings with and without prefetching, we can observe the benefit.
- The code checks for errors at each CUDA API call, prints the timings, and verifies that the computation result is correct.
- The example uses device 0 (or the first available GPU). If the system has multiple GPUs, the code can be adapted to prefetch to a specific GPU by passing its device ID to `cudaMemPrefetchAsync`.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Simple vector addition kernel
__global__ void vectorAdd(const float *a, const float *b, float *c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main(void) {
    // Determine the device to use (device 0)
    int device = 0;
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }
    if (device >= deviceCount) device = deviceCount - 1;
    CHECK(cudaSetDevice(device));

    const size_t N = 1 << 24; // 16M elements (~64 MB)
    size_t bytes = N * sizeof(float);

    // Allocate unified memory
    float *a, *b, *c;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&b, bytes));
    CHECK(cudaMallocManaged(&c, bytes));

    // Initialize arrays on host
    for (size_t i = 0; i < N; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    CHECK(cudaEventCreate(&startEvent));
    CHECK(cudaEventCreate(&stopEvent));

    // ----- Pre-fetch data to GPU before kernel -----
    // Record event before prefetch
    CHECK(cudaEventRecord(startEvent, 0));
    // Prefetch a, b, c to device
    CHECK(cudaMemPrefetchAsync(a, bytes, device, 0));
    CHECK(cudaMemPrefetchAsync(b, bytes, device, 0));
    CHECK(cudaMemPrefetchAsync(c, bytes, device, 0));
    // Wait for prefetch to finish
    CHECK(cudaEventRecord(stopEvent, 0));
    CHECK(cudaEventSynchronize(stopEvent));

    float prefetchtime_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&prefetchtime_ms, startEvent, stopEvent));

    // ----- Launch kernel -----
    CHECK(cudaEventRecord(startEvent, 0));
    size_t blockSize = 256;
    size_t gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(a, b, c, N);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stopEvent, 0));
    CHECK(cudaEventSynchronize(stopEvent));

    float kernelTime_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&kernelTime_ms, startEvent, stopEvent));

    // ----- Pre-fetch result back to CPU after kernel -----
    CHECK(cudaEventRecord(startEvent, 0));
    CHECK(cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId, 0));
    CHECK(cudaEventRecord(stopEvent, 0));
    CHECK(cudaEventSynchronize(stopEvent));

    float postPrefetchTime_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&postPrefetchTime_ms, startEvent, stopEvent));

    // Verify results
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        if (c[i] != 3.0f) {
            success = false;
            fprintf(stderr, "Mismatch at index %zu: %f\n", i, c[i]);
            break;
        }
    }

    printf("Vector addition of %zu elements completed.\n", N);
    printf("Data prefetch to GPU: %.3f ms\n", prefetchtime_ms);
    printf("Kernel execution:     %.3f ms\n", kernelTime_ms);
    printf("Data prefetch to CPU: %.3f ms\n", postPrefetchTime_ms);
    printf("Computation %s.\n", success ? "succeeded" : "failed");

    // Clean up
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(c));
    CHECK(cudaEventDestroy(startEvent));
    CHECK(cudaEventDestroy(stopEvent));

    return 0;
}
```