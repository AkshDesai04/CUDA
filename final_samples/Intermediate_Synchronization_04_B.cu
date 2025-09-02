```cpp
/*
Time the naive (racy) kernel vs. the atomicAdd kernel. Atomic operations have a performance cost. Is it significant here?

To explore the performance cost of atomic operations we implement two simple CUDA kernels that increment a single
integer value many times. In the *naive* kernel each thread performs an ordinary read–modify–write sequence
`*value += 1;` without any synchronization. This introduces a data race: all threads write to the same memory
location concurrently, and the final value will be corrupted (many increments are lost). The *atomic* kernel
uses `atomicAdd(value, 1)` to safely serialize updates, guaranteeing that every increment is counted.
 
Although atomic operations incur overhead due to contention and serialization, the question is whether that
overhead is significant for a workload where each thread performs a modest number of increments. To answer this
we:
 
1. Allocate two device integers (`d_naive_val` and `d_atomic_val`) and initialize them to zero.
2. Launch both kernels with the same grid and block dimensions. Each thread performs a fixed number of
   increments (`inc_per_thread`), so the total work is identical for both kernels. We choose a value that
   keeps the final counter within 32‑bit range (e.g., `inc_per_thread = 1000` with 1024 threads per block
   and 256 blocks gives ~262 million increments).
3. Measure execution time using CUDA events (`cudaEventRecord`, `cudaEventElapsedTime`).
4. Copy back the results to host memory, compare the actual counter values to the expected value,
   and print timings along with the overhead percentage.
5. The code includes a simple `CHECK_CUDA` macro for error handling, and it can be compiled with
   `nvcc -arch=sm_61 program.cu -o program`.
 
The expectation is that the atomic kernel will be slower than the naive kernel, but the overhead may be
small relative to the total work if the number of increments per thread is not extremely high. By comparing
the timings we can quantify the impact of atomic operations in this particular scenario.
*/

#include <cstdio>
#include <cuda_runtime.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Naive kernel: each thread performs inc_per_thread increments on the same integer without atomicity.
__global__ void naiveKernel(int *value, int inc_per_thread) {
    // Every thread performs the same loop; all contend for the same memory location.
    for (int i = 0; i < inc_per_thread; ++i) {
        (*value) += 1;  // racy read-modify-write
    }
}

// Atomic kernel: each thread performs inc_per_thread atomic adds.
__global__ void atomicKernel(int *value, int inc_per_thread) {
    for (int i = 0; i < inc_per_thread; ++i) {
        atomicAdd(value, 1);  // safe concurrent increment
    }
}

int main(void) {
    const int threadsPerBlock = 1024;
    const int blocksPerGrid    = 256;
    const int inc_per_thread   = 1000;  // increments per thread

    const long long expected_total =
        static_cast<long long>(inc_per_thread) * threadsPerBlock * blocksPerGrid;

    int *d_naive_val = nullptr;
    int *d_atomic_val = nullptr;
    int h_naive_val = 0;
    int h_atomic_val = 0;

    // Allocate device memory for counters
    CHECK_CUDA(cudaMalloc((void**)&d_naive_val, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_atomic_val, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_naive_val, 0, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_atomic_val, 0, sizeof(int)));

    // CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ----------------- Naive kernel -----------------
    CHECK_CUDA(cudaEventRecord(start, 0));
    naiveKernel<<<blocksPerGrid, threadsPerBlock>>>(d_naive_val, inc_per_thread);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float timeNaive = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&timeNaive, start, stop));

    // ----------------- Atomic kernel -----------------
    CHECK_CUDA(cudaEventRecord(start, 0));
    atomicKernel<<<blocksPerGrid, threadsPerBlock>>>(d_atomic_val, inc_per_thread);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float timeAtomic = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&timeAtomic, start, stop));

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(&h_naive_val, d_naive_val, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_atomic_val, d_atomic_val, sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Expected total increments: %lld\n", expected_total);
    printf("Naive kernel result:       %d\n", h_naive_val);
    printf("Atomic kernel result:      %d\n", h_atomic_val);
    printf("\n");
    printf("Naive kernel time:  %.3f ms\n", timeNaive);
    printf("Atomic kernel time: %.3f ms\n", timeAtomic);
    printf("\n");
    float overhead = ((timeAtomic - timeNaive) / timeNaive) * 100.0f;
    printf("Atomic vs naive overhead: %.2f%%\n", overhead);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_naive_val));
    CHECK_CUDA(cudaFree(d_atomic_val));

    return 0;
}
```