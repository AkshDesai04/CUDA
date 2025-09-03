```cpp
/*
An alternative to loading 0 is to make the computation loop conditional. Is this more or less efficient?

In this program we compare two kernel implementations that compute a simple sum over an array of integers. 
Both kernels ultimately perform the same arithmetic, but they differ in how they handle the initial
value of the accumulator used in the inner loop:

1. **kernelLoadZero** – The accumulator is initialized to 0 unconditionally before the loop.
2. **kernelConditional** – The loop (and the accumulator initialization) is executed only when a 
   runtime condition (`loopCount > 0`) is true. If `loopCount` is zero, the kernel does nothing.

The intention is to explore whether avoiding the explicit load of zero can be more efficient than
always loading zero and then possibly skipping the loop body. We expect the conditional approach to
incur a branch penalty that may outweigh the benefit of saving one memory load, especially when
`loopCount` is small. The program measures the execution time of each kernel for various
`loopCount` values and reports the results.

We use CUDA events for timing, launch both kernels multiple times for a stable measurement, 
and print the average execution times in microseconds. The kernels are deliberately simple to
avoid other confounding factors, allowing a clear comparison of the two strategies.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

__global__ void kernelLoadZero(const int *data, int *result, int dataSize, int loopCount)
{
    // Each thread processes a distinct element of the input array
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dataSize) return;

    // Accumulator initialized to 0 unconditionally
    int sum = 0;
    for (int i = 0; i < loopCount; ++i) {
        // Simple operation that depends on the loop count
        sum += data[idx] + i;
    }
    result[idx] = sum;
}

__global__ void kernelConditional(const int *data, int *result, int dataSize, int loopCount)
{
    // Each thread processes a distinct element of the input array
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dataSize) return;

    if (loopCount > 0) {
        // Accumulator and loop executed only when loopCount > 0
        int sum = 0;
        for (int i = 0; i < loopCount; ++i) {
            sum += data[idx] + i;
        }
        result[idx] = sum;
    } else {
        // When loopCount == 0, no operations are performed
        result[idx] = 0; // optional: leave unchanged or set to zero
    }
}

void runBenchmark(int dataSize, int loopCount)
{
    int threadsPerBlock = 256;
    int blocks = (dataSize + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    int *h_input = (int*)malloc(dataSize * sizeof(int));
    int *h_output = (int*)malloc(dataSize * sizeof(int));

    // Initialize input data
    for (int i = 0; i < dataSize; ++i) {
        h_input[i] = i % 100; // arbitrary pattern
    }

    // Allocate device memory
    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input, dataSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, dataSize * sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, dataSize * sizeof(int), cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int runs = 10;
    float totalTimeLoadZero = 0.0f;
    float totalTimeConditional = 0.0f;

    // Benchmark kernelLoadZero
    for (int r = 0; r < runs; ++r) {
        CUDA_CHECK(cudaEventRecord(start, 0));
        kernelLoadZero<<<blocks, threadsPerBlock>>>(d_input, d_output, dataSize, loopCount);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        totalTimeLoadZero += ms;
    }

    // Benchmark kernelConditional
    for (int r = 0; r < runs; ++r) {
        CUDA_CHECK(cudaEventRecord(start, 0));
        kernelConditional<<<blocks, threadsPerBlock>>>(d_input, d_output, dataSize, loopCount);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        totalTimeConditional += ms;
    }

    // Copy result back (optional)
    CUDA_CHECK(cudaMemcpy(h_output, d_output, dataSize * sizeof(int), cudaMemcpyDeviceToHost));

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    printf("=== Benchmark Results ===\n");
    printf("Data size: %d elements\n", dataSize);
    printf("Loop count: %d\n", loopCount);
    printf("Average time over %d runs (ms):\n", runs);
    printf("  kernelLoadZero    : %6.3f ms\n", totalTimeLoadZero / runs);
    printf("  kernelConditional : %6.3f ms\n", totalTimeConditional / runs);
    if (totalTimeLoadZero < totalTimeConditional) {
        printf("Result: kernelLoadZero is faster.\n");
    } else {
        printf("Result: kernelConditional is faster.\n");
    }
    printf("==========================\n\n");
}

int main()
{
    // Test with different loop counts
    runBenchmark(1 << 20, 0);    // loopCount = 0
    runBenchmark(1 << 20, 10);   // loopCount = 10
    runBenchmark(1 << 20, 1000); // loopCount = 1000

    return 0;
}
```