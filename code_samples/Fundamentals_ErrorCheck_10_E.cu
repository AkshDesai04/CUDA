```cpp
/*
 * Aim of the program: Use the timer to time a `cudaMemcpy` operation.
 *
 * Thinking:
 * The task is to measure how long a single `cudaMemcpy` call takes.  
 * CUDA provides high‑resolution timing through event objects (`cudaEvent_t`).  
 * We'll create two events, record them before and after the `cudaMemcpy`, and then
 * compute the elapsed time with `cudaEventElapsedTime`.  This gives the time in
 * milliseconds.  
 *
 * The program will:
 * 1. Allocate a large host array (e.g., 10 million integers) and initialize it.
 * 2. Allocate a device array of the same size.
 * 3. Create two events (`start` and `stop`).
 * 4. Record the `start` event, perform a host‑to‑device `cudaMemcpy`,
 *    then record the `stop` event.
 * 5. Synchronize the device, calculate the elapsed time, and print it.
 * 6. Clean up all allocated memory and events.
 *
 * For safety, error checking macros (`checkCudaErrors`) are used to catch
 * any CUDA API failures.  The program is self‑contained and can be compiled
 * with `nvcc` (e.g., `nvcc -o cudaMemcpyTimer cudaMemcpyTimer.cu`).
 *
 * The key focus is on using the CUDA event timer to benchmark the
 * data transfer speed, which is critical in GPU‑accelerated applications.
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Utility macro for checking CUDA API calls */
#define checkCudaErrors(val) checkCudaErrorsImpl((val), #val, __FILE__, __LINE__)

inline void checkCudaErrorsImpl(cudaError_t err, const char *func, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, static_cast<int>(err), cudaGetErrorName(err), func);
        exit(EXIT_FAILURE);
    }
}

int main() {
    const size_t N = 10'000'000;          // Number of integers
    const size_t size = N * sizeof(int); // Size in bytes

    /* Host memory allocation */
    int *h_array = (int*)malloc(size);
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host array */
    for (size_t i = 0; i < N; ++i) {
        h_array[i] = static_cast<int>(i);
    }

    /* Device memory allocation */
    int *d_array = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_array, size));

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    /* Record start event */
    checkCudaErrors(cudaEventRecord(start, 0));

    /* Perform the memcpy from host to device */
    checkCudaErrors(cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice));

    /* Record stop event */
    checkCudaErrors(cudaEventRecord(stop, 0));

    /* Wait for the event to complete */
    checkCudaErrors(cudaEventSynchronize(stop));

    /* Compute elapsed time */
    float milliseconds = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("cudaMemcpy (host -> device) of %zu MB took %.3f ms\n",
           size / (1024 * 1024), milliseconds);

    /* Clean up */
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFree(d_array));
    free(h_array);

    return EXIT_SUCCESS;
}
```