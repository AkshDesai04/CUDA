```cuda
/*
Write a custom kernel `void copy_kernel(int* in, int* out, int N)` that performs the same copy operation as `cudaMemcpyDeviceToDevice`. Time this kernel and compare its performance to the API call.

Thinking:
- We need a kernel that copies elements from one device array to another.
- Kernel signature: `__global__ void copy_kernel(int* in, int* out, int N)`.
- Each thread will compute a global index and, if within bounds, copy `in[idx]` to `out[idx]`.
- On the host side, we will:
  1. Allocate two device buffers (`d_in` and `d_out`).
  2. Initialize `d_in` with some values (e.g., sequential integers) using a simple kernel or `cudaMemcpyFromSymbol`/`cudaMemcpyToArray`. We'll use a small helper kernel `init_kernel`.
  3. Time `cudaMemcpyDeviceToDevice` with `cudaEventRecord`.
  4. Time the custom `copy_kernel` similarly.
  5. Compare elapsed times.
  6. Verify correctness by copying results back to host and checking equality.
- For timing, use CUDA events (`cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`).
- Use a sufficiently large array size (e.g., `N = 1 << 24` ~ 16 million ints) to make the copy substantial.
- Use a reasonable block size (e.g., 256) and compute grid size accordingly.
- Include error checking macros to simplify debugging.
- Print out the times and the difference in performance.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),   \
                    cudaGetErrorString(err));                         \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Simple kernel to initialize an array with sequential values
__global__ void init_kernel(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) d_arr[idx] = idx;
}

// Custom copy kernel
__global__ void copy_kernel(const int *in, int *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = in[idx];
}

int main(void) {
    const int N = 1 << 24; // 16,777,216 elements (~64 MB for int)
    const size_t bytes = N * sizeof(int);

    // Allocate device memory
    int *d_in = nullptr;
    int *d_out_api = nullptr;
    int *d_out_kernel = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out_api, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out_kernel, bytes));

    // Initialize d_in
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    init_kernel<<<blocks, threadsPerBlock>>>(d_in, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing structures
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 1. Time cudaMemcpyDeviceToDevice
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(d_out_api, d_in, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_api = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_api, start, stop));

    // 2. Time custom copy kernel
    CHECK_CUDA(cudaEventRecord(start));
    copy_kernel<<<blocks, threadsPerBlock>>>(d_in, d_out_kernel, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_kernel = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_kernel, start, stop));

    // Copy results back to host to verify correctness
    int *h_out_api = (int*)malloc(bytes);
    int *h_out_kernel = (int*)malloc(bytes);
    CHECK_CUDA(cudaMemcpy(h_out_api, d_out_api, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_kernel, d_out_kernel, bytes, cudaMemcpyDeviceToHost));

    // Verify
    int mismatch = 0;
    for (int i = 0; i < N; ++i) {
        if (h_out_api[i] != h_out_kernel[i]) {
            ++mismatch;
            if (mismatch <= 10) {
                printf("Mismatch at %d: API=%d, Kernel=%d\n", i, h_out_api[i], h_out_kernel[i]);
            }
        }
    }
    if (mismatch == 0) {
        printf("Verification passed: all elements match.\n");
    } else {
        printf("Verification failed: %d mismatches.\n", mismatch);
    }

    // Print timing results
    printf("cudaMemcpyDeviceToDevice time: %.3f ms\n", ms_api);
    printf("Custom copy kernel time:       %.3f ms\n", ms_kernel);
    printf("Speedup (API / Kernel): %.2f\n", ms_api / ms_kernel);

    // Cleanup
    free(h_out_api);
    free(h_out_kernel);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out_api));
    CHECK_CUDA(cudaFree(d_out_kernel));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```