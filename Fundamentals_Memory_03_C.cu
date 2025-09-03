```cpp
/*
Write a custom kernel `void copy_kernel(int* in, int* out, int N)` that performs the same copy operation as `cudaMemcpyDeviceToDevice`. Time this kernel and compare its performance to the API call.

Thought process:
1. The goal is to demonstrate that a user‑defined kernel can copy data on the device just like the built‑in cudaMemcpyDeviceToDevice, and to see how the timings compare.
2. We need a simple element‑wise copy kernel:
      __global__ void copy_kernel(int *in, int *out, int N) {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          if (idx < N) out[idx] = in[idx];
      }
3. Timing is done with CUDA events (cudaEvent_t). For each operation we create start and stop events, record the start event, launch the operation (kernel or cudaMemcpy), record the stop event, synchronize, and then query elapsed time.
4. To compare, we allocate a source array and a destination array on the device. We copy data from host to source to set up the test.
5. After running the kernel copy we copy the destination back to host and verify that every element matches the source. Then we reset the destination (e.g., to a distinct value with cudaMemset).
6. Then we perform the API copy (cudaMemcpyDeviceToDevice) from source to destination, time it, copy destination back to host, and verify again.
7. Finally we print the measured times. We expect the API copy to be faster for large contiguous data because it is highly optimized, while the kernel incurs launch overhead but can be beneficial if combined with other operations or when the copy is part of a more complex kernel launch.
8. All CUDA API calls are wrapped in an error‑checking macro to ensure we catch any issues early.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// Kernel performing element‑wise copy
__global__ void copy_kernel(const int *in, int *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx];
    }
}

int main(void) {
    const int N = 1 << 20;  // 1M integers
    const size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_src = (int*)malloc(size);
    int *h_dst_kernel = (int*)malloc(size);
    int *h_dst_memcpy = (int*)malloc(size);

    // Initialize source with deterministic data
    for (int i = 0; i < N; ++i) {
        h_src[i] = i;
    }

    // Allocate device memory
    int *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc((void**)&d_src, size));
    CHECK_CUDA(cudaMalloc((void**)&d_dst, size));

    // Copy source data to device
    CHECK_CUDA(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice));

    // ------------------- Kernel copy -------------------
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Timing kernel
    CHECK_CUDA(cudaEventRecord(start));
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    copy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float kernelTime = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernelTime, start, stop));

    // Copy result back to host and verify
    CHECK_CUDA(cudaMemcpy(h_dst_kernel, d_dst, size, cudaMemcpyDeviceToHost));
    int correct = 1;
    for (int i = 0; i < N; ++i) {
        if (h_dst_kernel[i] != h_src[i]) {
            correct = 0;
            printf("Mismatch at index %d: src=%d, dst=%d\n", i, h_src[i], h_dst_kernel[i]);
            break;
        }
    }
    printf("Kernel copy %s.\n", correct ? "PASSED" : "FAILED");

    // Reset destination buffer to a known bad value
    CHECK_CUDA(cudaMemset(d_dst, 0xFF, size));

    // ------------------- API copy -------------------
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float apiTime = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&apiTime, start, stop));

    // Copy result back to host and verify
    CHECK_CUDA(cudaMemcpy(h_dst_memcpy, d_dst, size, cudaMemcpyDeviceToHost));
    correct = 1;
    for (int i = 0; i < N; ++i) {
        if (h_dst_memcpy[i] != h_src[i]) {
            correct = 0;
            printf("Mismatch at index %d: src=%d, dst=%d\n", i, h_src[i], h_dst_memcpy[i]);
            break;
        }
    }
    printf("API copy %s.\n", correct ? "PASSED" : "FAILED");

    // Print timings
    printf("Kernel time: %.3f ms\n", kernelTime);
    printf("API copy time: %.3f ms\n", apiTime);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    free(h_src);
    free(h_dst_kernel);
    free(h_dst_memcpy);

    return 0;
}
```