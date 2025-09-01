/*
Aim: Write a custom kernel void copy_kernel(int* in, int* out, int N) that performs the same copy operation as cudaMemcpyDeviceToDevice. Time this kernel and compare its performance to the API call.

Thinking: 
1. The kernel should iterate over all indices < N, copying each element from the input array to the output array.
2. We'll use a typical 256-thread block configuration, launching enough blocks to cover N elements.
3. Timing will be performed with CUDA events (cudaEvent_t) for high-resolution GPU timing.
4. Device memory for the input and output arrays will be allocated with cudaMalloc. The input array will be filled with dummy data using cudaMemset to avoid the overhead of a host-to-device transfer.
5. The kernel execution will be timed: record start, launch kernel, record stop, sync, then compute elapsed time.
6. The cudaMemcpyDeviceToDevice operation will be timed in a similar fashion, using events around the call.
7. Both timings will be printed. Optionally, correctness could be verified by copying the output back to host and checking a few elements, but this is omitted for brevity.
8. An error-checking macro is provided to simplify CUDA error handling.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",             \
                    __FUNCTION__, __FILE__, __LINE__,                    \
                    cudaGetErrorString(err));                            \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// Custom copy kernel
__global__ void copy_kernel(const int* in, int* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx];
    }
}

int main(void) {
    const int N = 1 << 24; // Number of integers (≈16M)
    size_t size = N * sizeof(int);

    int *d_in = NULL;
    int *d_out = NULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_in, size));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size));

    // Fill input array with dummy data (e.g., all 0xFF)
    CUDA_CHECK(cudaMemset(d_in, 0xFF, size));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ==========================
    // Time custom kernel copy
    // ==========================
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaEventRecord(start, 0));
    copy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for kernel to finish
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));
    printf("Custom kernel copy time: %f ms\n", kernelTimeMs);

    // ==========================
    // Time cudaMemcpyDeviceToDevice
    // ==========================
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(d_out, d_in, size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for copy to finish
    CUDA_CHECK(cudaEventSynchronize(stop));

    float apiTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&apiTimeMs, start, stop));
    printf("cudaMemcpyDeviceToDevice time: %f ms\n", apiTimeMs);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
