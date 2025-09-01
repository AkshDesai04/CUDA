```cpp
/*
Aim of the program:
Write a custom kernel `void copy_kernel(int* in, int* out, int N)` that performs the same copy operation as `cudaMemcpyDeviceToDevice`. Time this kernel and compare its performance to the API call.

Thinking:
1. Define the kernel that copies each element from `in` to `out`. Use a 1D grid of threads, each thread copying one element if its global index is within bounds.
2. In `main`, allocate host memory for an array of size N, initialize it, and allocate two device buffers `d_in` and `d_out`.
3. Copy the host array to `d_in` using `cudaMemcpyHostToDevice`.
4. Time the custom kernel:
   - Create CUDA events for start and stop.
   - Record start event, launch the kernel with an appropriate block size (e.g., 256 threads per block), record stop event, and synchronize.
   - Compute elapsed time in milliseconds.
5. Time the built-in `cudaMemcpyDeviceToDevice`:
   - Record start event, perform `cudaMemcpy` from `d_in` to `d_out` with `cudaMemcpyDeviceToDevice`, record stop event, synchronize.
   - Compute elapsed time.
6. Copy the result back to host from both `d_out` copies (one from kernel, one from API) to verify correctness. Since we overwrite `d_out` after each copy, we will reuse the same buffer and check the data immediately after each operation.
7. Print both times and the relative performance.
8. Include error checking macros to make debugging easier.
9. Ensure that all allocated resources are freed before exiting.

The program will be a self‑contained .cu file with all necessary includes and definitions. It demonstrates how to benchmark a custom device-to-device copy kernel against the official CUDA API call.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Kernel that copies each element from in to out
__global__ void copy_kernel(int *in, int *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx];
    }
}

int main(void) {
    const int N = 1 << 24; // 16M elements (~64MB for int array)
    size_t bytes = N * sizeof(int);

    // Allocate and initialize host memory
    int *h_in = (int*)malloc(bytes);
    int *h_out_cpu = (int*)malloc(bytes); // for host verification
    assert(h_in && h_out_cpu);

    for (int i = 0; i < N; ++i) {
        h_in[i] = i;
    }

    // Allocate device memory
    int *d_in = nullptr;
    int *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // ---------- Timing custom kernel ----------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Record start
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    copy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch errors

    // Record stop
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    // Copy result back to host for verification
    CUDA_CHECK(cudaMemcpy(h_out_cpu, d_out, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness
    for (int i = 0; i < N; ++i) {
        if (h_out_cpu[i] != h_in[i]) {
            fprintf(stderr, "Data mismatch at index %d: got %d, expected %d\n",
                    i, h_out_cpu[i], h_in[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("Custom kernel copy succeeded in %.3f ms\n", kernel_ms);

    // ---------- Timing cudaMemcpyDeviceToDevice ----------
    // Record start
    CUDA_CHECK(cudaEventRecord(start, 0));

    CUDA_CHECK(cudaMemcpy(d_out, d_in, bytes, cudaMemcpyDeviceToDevice));

    // Record stop
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float api_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&api_ms, start, stop));

    // Copy result back to host for verification
    CUDA_CHECK(cudaMemcpy(h_out_cpu, d_out, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness
    for (int i = 0; i < N; ++i) {
        if (h_out_cpu[i] != h_in[i]) {
            fprintf(stderr, "Data mismatch after cudaMemcpyDeviceToDevice at index %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("cudaMemcpyDeviceToDevice succeeded in %.3f ms\n", api_ms);

    // Print comparison
    printf("\nPerformance comparison:\n");
    printf("Kernel time: %.3f ms\n", kernel_ms);
    printf("API time  : %.3f ms\n", api_ms);
    if (kernel_ms < api_ms)
        printf("Kernel is faster by %.2fx\n", api_ms / kernel_ms);
    else
        printf("API is faster by %.2fx\n", kernel_ms / api_ms);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out_cpu);

    return 0;
}
```