/*
Compare the performance of `cudaMemset` vs. your custom zeroing kernel for a very large array. `cudaMemset` is usually highly optimized.

Thinking:
The goal is to give the user a minimal but complete CUDA program that measures and compares the time taken by the highly optimized built‑in `cudaMemset` versus a simple user‑written kernel that sets every element of a large array to zero.  
Key design choices:

1. **Array size** – To exercise the GPU bandwidth we use a large 1‑D array of floats.  
   `1 << 26` (67 108 864) elements equal 256 MiB. This is big enough for a noticeable
   bandwidth figure on typical GPUs but small enough to fit on a wide range of
   devices.

2. **Timing** – CUDA events (`cudaEventCreate`, `cudaEventRecord`, `cudaEventElapsedTime`)
   provide GPU‑side timing with nanosecond resolution and are inexpensive.

3. **Kernel launch configuration** – A 256‑thread block is a common choice that
   works well on many GPUs. The grid size is computed as
   `(N + blockDim - 1) / blockDim` to cover all elements.

4. **Correctness verification** – After each operation we copy a small portion
   back to host memory and check that the first element is zero.  This
   protects against a compiler optimization that could otherwise eliminate
   the kernel.

5. **Error checking** – A simple `CUDA_CHECK` macro wraps CUDA API calls
   to abort on failure and print the error string.

6. **Output** – The program prints the elapsed time for `cudaMemset` and the
   custom kernel in milliseconds, giving the user an easy comparison.

The code is self‑contained, written in CUDA C, and can be compiled with
`nvcc zero_comparison.cu -o zero_comparison`.  Running the binary will
print the timings to the console.  Feel free to adjust `ARRAY_SIZE` if
you have a different GPU memory budget.

*/

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Custom kernel to set each element of an array to zero
__global__ void zeroKernel(float *d_arr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = 0.0f;
    }
}

int main() {
    const size_t ARRAY_SIZE = 1 << 26; // 67,108,864 floats (~256 MB)
    const size_t ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    float *d_arr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_arr, ARRAY_BYTES));

    // Host buffer for verification (small)
    float h_check[4];

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 1. Measure cudaMemset
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemset(d_arr, 0, ARRAY_BYTES));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float memset_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&memset_ms, start, stop));
    printf("cudaMemset time: %.3f ms\n", memset_ms);

    // Verify (copy back first element)
    CUDA_CHECK(cudaMemcpy(h_check, d_arr, sizeof(h_check), cudaMemcpyDeviceToHost));
    printf("After cudaMemset, first element: %f\n", h_check[0]);

    // 2. Measure custom zeroing kernel
    const int THREADS_PER_BLOCK = 256;
    int blocks = (ARRAY_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    CUDA_CHECK(cudaEventRecord(start, 0));
    zeroKernel<<<blocks, THREADS_PER_BLOCK>>>(d_arr, ARRAY_SIZE);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch error
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    printf("Custom zero kernel time: %.3f ms\n", kernel_ms);

    // Verify again
    CUDA_CHECK(cudaMemcpy(h_check, d_arr, sizeof(h_check), cudaMemcpyDeviceToHost));
    printf("After custom kernel, first element: %f\n", h_check[0]);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_arr));

    return 0;
}
