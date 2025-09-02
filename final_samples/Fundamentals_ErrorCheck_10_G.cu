```cpp
/*
What happens if you call `elapsed_ms()` before `stop()` has been called or before the event has completed? The result is undefined or an error.

Thinking:
- The CUDA API provides cudaEventElapsedTime to compute elapsed time between two events.
- It requires that both events have been recorded with cudaEventRecord before the call.
- If the stop event has not been recorded yet, or the events have not been properly synchronized, cudaEventElapsedTime may return cudaErrorNotReady or other error, indicating the operation cannot be performed.
- Calling cudaEventElapsedTime before stop is recorded could lead to undefined behavior if the API does not guard against it, but in practice it returns an error.
- The demonstration will record a start event, launch a kernel, call cudaEventElapsedTime immediately (before recording stop), print the error, then record stop, and call again to get a valid elapsed time.
- This shows that the result is undefined or error until the stop event has completed.
*/
#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                     \
    } while (0)

// A simple kernel that performs a small amount of work
__global__ void dummyKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple computation to keep the GPU busy
        for (int i = 0; i < 1000; ++i) {
            data[idx] = data[idx] * (i % 3 + 1);
        }
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_data = (int *)malloc(size);
    for (int i = 0; i < N; ++i) h_data[i] = i;

    // Allocate device memory
    int *d_data;
    CHECK_CUDA(cudaMalloc((void **)&d_data, size));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Create CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dummyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    // **Important:** We do NOT record the stop event yet.
    // Call cudaEventElapsedTime before stop has been recorded.
    float elapsed_ms = 0.0f;
    cudaError_t err = cudaEventElapsedTime(&elapsed_ms, start, stop);
    if (err != cudaSuccess) {
        printf("First cudaEventElapsedTime call (before stop recorded) returned error: %s\n",
               cudaGetErrorString(err));
    } else {
        printf("First cudaEventElapsedTime call (before stop recorded) succeeded unexpectedly. Elapsed: %f ms\n",
               elapsed_ms);
    }

    // Now record the stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Now call cudaEventElapsedTime again, should succeed
    err = cudaEventElapsedTime(&elapsed_ms, start, stop);
    if (err != cudaSuccess) {
        printf("Second cudaEventElapsedTime call failed: %s\n",
               cudaGetErrorString(err));
    } else {
        printf("Second cudaEventElapsedTime call succeeded. Elapsed: %f ms\n",
               elapsed_ms);
    }

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);

    return 0;
}
```