```c
/*
Aim: Perform the same timing for a device-to-device copy.

Thinking:
- The task is to time a CUDA device-to-device memory copy operation.
- Allocate two arrays on the device: src and dst.
- Initialize src from host (e.g., fill with some values) and copy it to the device.
- Use CUDA events (cudaEventCreate, cudaEventRecord, cudaEventSynchronize,
  cudaEventElapsedTime) to measure the duration of cudaMemcpy from src to dst
  with cudaMemcpyDeviceToDevice.
- After the timed copy, optionally copy dst back to host to verify correctness.
- Print the elapsed time in milliseconds.
- Include basic error checking for CUDA API calls.
- The program is self‑contained, written in C/CUDA, and can be compiled with nvcc.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Macro for checking CUDA API call results
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    const size_t N = 1 << 20;          // 1M elements
    const size_t size_bytes = N * sizeof(float);

    // Allocate host memory and initialize
    float *h_src = (float*)malloc(size_bytes);
    float *h_dst = (float*)malloc(size_bytes);
    if (!h_src || !h_dst) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < N; ++i) {
        h_src[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_src = nullptr;
    float *d_dst = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_src, size_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_dst, size_bytes));

    // Copy source data from host to device
    CHECK_CUDA(cudaMemcpy(d_src, h_src, size_bytes, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Perform device-to-device copy
    CHECK_CUDA(cudaMemcpy(d_dst, d_src, size_bytes, cudaMemcpyDeviceToDevice));

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Compute elapsed time
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("Device-to-device copy of %zu floats took %f ms\n", N, ms);

    // Optional: Copy back to host to verify
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, size_bytes, cudaMemcpyDeviceToHost));
    // Simple verification: check first few elements
    bool ok = true;
    for (size_t i = 0; i < 10; ++i) {
        if (h_src[i] != h_dst[i]) {
            printf("Mismatch at index %zu: src=%f dst=%f\n", i, h_src[i], h_dst[i]);
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("Verification passed: first 10 elements match.\n");
    }

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    free(h_src);
    free(h_dst);

    return EXIT_SUCCESS;
}
```