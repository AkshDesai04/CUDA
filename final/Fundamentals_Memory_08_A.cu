```cuda
/* 
Perform the same timing for a device-to-host copy. Is it typically faster, slower, or the same?

I want to write a simple CUDA program that measures how long it takes to copy data
from device memory to host memory using cudaMemcpy with cudaMemcpyDeviceToHost.  
The aim is to use CUDA events to time the transfer, which gives us GPU-side
timing that is accurate for the actual data transfer.  
The program will:
  1. Allocate a large array on the host and the device (e.g., 100 million floats).
  2. Initialize the device array with some data (e.g., fill with 1.0f).
  3. Perform a warm‑up copy from device to host so that any initial
     overheads are amortized.
  4. Run several iterations of device‑to‑host copies, measuring the elapsed
     time for each iteration with cudaEventRecord and cudaEventElapsedTime.
  5. Compute and print the average copy time in milliseconds.
  6. Clean up allocated memory.

I’ll also include error checking after each CUDA call, so that any failures
are reported clearly.  The code will be minimal but fully functional and
self‑contained, suitable for compiling with nvcc. 
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",     \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    const size_t N = 100 * 1024 * 1024; // 100 million floats (~400 MB)
    const int iterations = 10;

    // Host memory allocation
    float *h_data = (float*)malloc(N * sizeof(float));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Device memory allocation
    float *d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, N * sizeof(float)));

    // Initialize device array to some value (e.g., 1.0f)
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float))); // zero for simplicity

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up copy: device to host (ignored in timing)
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Timing loop
    float totalTime = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start, 0));

        CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime += milliseconds;

        printf("Iteration %2d: %f ms\n", i + 1, milliseconds);
    }

    float avgTime = totalTime / iterations;
    printf("\nAverage device-to-host copy time: %f ms\n", avgTime);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return 0;
}
```