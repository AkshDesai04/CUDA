```cpp
/*
Compare the performance of the shared memory reversal to a naive kernel where `output[i] = input[N-1-i]` for a large array N.

The aim of this program is to benchmark two CUDA kernels that reverse a large array.  
The naive kernel simply writes each element to its mirrored position in the output array.  
The shared‑memory kernel first loads a contiguous block of the input array into shared memory
and then writes the reversed elements from shared memory to the global output array.  
We measure the execution time of each kernel on a large array (N = 2^26) and compare the
results.  The program also verifies the correctness of the reversal by checking a few
sample elements after each kernel execution.

The design decisions are:
1. Use `float` arrays to keep memory usage moderate (~268 MB per array).
2. Allocate host and device buffers, initialize the host input with a simple sequence.
3. Use `cudaEvent_t` for timing, performing `cudaDeviceSynchronize()` after each kernel to
   ensure accurate measurements.
4. The naive kernel uses one global read and one global write per element.
5. The shared‑memory kernel also uses one global read per element, plus one write from
   shared memory; this may improve memory coalescing for the read but adds shared memory
   usage.
6. For the shared kernel, we guard all accesses with `idx < N` to handle the last block
   that may be incomplete.
7. The program prints the elapsed times and the speedup factor.
8. The correctness is verified by comparing a few elements of the output to the expected
   reversed values.

This file is a complete, compilable CUDA C program and can be built with:
    nvcc -O2 -arch=sm_70 -o reverse_benchmark reverse_benchmark.cu
and executed directly.
*/
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",     \
                    __FILE__, __LINE__, static_cast<int>(err),               \
                    cudaGetErrorName(err), cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel: Naive reversal
__global__ void reverse_naive(const float *in, float *out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[N - 1 - idx] = in[idx];
    }
}

// Kernel: Shared memory reversal
__global__ void reverse_shared(const float *in, float *out, int N)
{
    extern __shared__ float s_mem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    if (idx < N) {
        s_mem[localIdx] = in[idx];
    }
    __syncthreads();

    if (idx < N) {
        // The element that belongs to this thread in the reversed order
        int revLocalIdx = blockDim.x - 1 - localIdx;
        out[N - 1 - idx] = s_mem[revLocalIdx];
    }
}

int main()
{
    const int N = 1 << 26;          // 67,108,864 elements (~268 MB for float)
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate host memory
    float *h_in  = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)malloc(N * sizeof(float));
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_in  = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_out, N * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Naive kernel timing
    CUDA_CHECK(cudaEventRecord(start));
    reverse_naive<<<GRID_SIZE, BLOCK_SIZE>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_naive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));

    // Copy result back and verify correctness
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < 10; ++i) {
        if (h_out[i] != static_cast<float>(N - 1 - i)) {
            correct = false;
            break;
        }
    }
    if (!correct) {
        fprintf(stderr, "Naive kernel produced incorrect result\n");
        return EXIT_FAILURE;
    }

    printf("Naive kernel time: %.3f ms\n", ms_naive);

    // Shared memory kernel timing
    size_t sharedMemSize = BLOCK_SIZE * sizeof(float);
    CUDA_CHECK(cudaEventRecord(start));
    reverse_shared<<<GRID_SIZE, BLOCK_SIZE, sharedMemSize>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_shared = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_shared, start, stop));

    // Copy result back and verify correctness
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    correct = true;
    for (int i = 0; i < 10; ++i) {
        if (h_out[i] != static_cast<float>(N - 1 - i)) {
            correct = false;
            break;
        }
    }
    if (!correct) {
        fprintf(stderr, "Shared memory kernel produced incorrect result\n");
        return EXIT_FAILURE;
    }

    printf("Shared memory kernel time: %.3f ms\n", ms_shared);
    if (ms_shared > 0.0f)
        printf("Speedup (naive / shared): %.2f\n", ms_naive / ms_shared);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return EXIT_SUCCESS;
}
```