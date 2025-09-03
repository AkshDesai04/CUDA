```c
/* 
Use the timer to time a `cudaMemcpy` operation.

Thinking process:
- The goal is to demonstrate how to measure the duration of a CUDA memory copy
  operation using CUDA’s own timing primitives, specifically cudaEvent_t.
- The program will:
  1. Allocate a small array on the host.
  2. Allocate a corresponding array on the device.
  3. Create two CUDA events: one before the copy and one after.
  4. Record the start event, perform cudaMemcpy (e.g., HostToDevice),
     record the end event, then synchronize.
  5. Compute the elapsed time between the events.
  6. Print the elapsed time in milliseconds.
- Error checking will be minimal but sufficient to catch allocation and
  copy errors.
- The code is self‑contained and can be compiled with nvcc:
    nvcc -o cuda_memcpy_timer cuda_memcpy_timer.cu
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error-checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",   \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),        \
                    cudaGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

int main(void) {
    const size_t N = 1 << 20;           // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_ptr = (float*)malloc(bytes);
    if (!h_ptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    // Initialize host data
    for (size_t i = 0; i < N; ++i) h_ptr[i] = (float)i;

    // Allocate device memory
    float *d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, bytes));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record the start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Perform the memory copy (Host to Device)
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));

    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for the copy to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time in milliseconds
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Output the timing result
    printf("cudaMemcpy (HostToDevice) time: %f ms\n", milliseconds);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_ptr));
    free(h_ptr);

    return EXIT_SUCCESS;
}
```