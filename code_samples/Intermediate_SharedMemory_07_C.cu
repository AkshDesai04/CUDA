```cpp
/* 
Compare the performance of the shared memory reversal to a naive kernel where `output[i] = input[N-1-i]` for a large array N.

Thinking:
- We need to implement two CUDA kernels that reverse a large array of floats.
- The naive kernel simply writes each output element by directly indexing the input in reverse order:
    output[gid] = input[N-1-gid];
  This results in non-coalesced writes because output indices are decreasing as gid increases.
- The shared memory kernel aims to improve write coalescing by loading the reversed elements into shared memory
  and then writing them to contiguous output positions:
    1. Each thread calculates the source index src = N-1-gid.
    2. It reads input[src] into shared[tid].
    3. After __syncthreads(), each thread writes shared[tid] to output[gid].
  This ensures contiguous global writes while still reading the same non-coalesced input pattern.
- Performance measurement is done using CUDA events for accurate timing.
- The program will:
    1. Allocate a large array on the host, fill it with random floats.
    2. Copy to device.
    3. Run both kernels multiple times to average timing.
    4. Verify correctness by comparing the device output to a reference reversed array.
- We use block size 256 threads; array size is 1 << 26 (~67 million) for a large test.
- Error checking macros are defined for clean code.
- The program is written in C/C++ CUDA and compiles to a .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

__global__ void reverse_naive(const float *input, float *output, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    output[gid] = input[N - 1 - gid];
}

__global__ void reverse_shared(const float *input, float *output, int N) {
    extern __shared__ float sh[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    int src = N - 1 - gid;
    sh[threadIdx.x] = input[src];
    __syncthreads();
    output[gid] = sh[threadIdx.x];
}

void generate_random_array(float *arr, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        arr[i] = (float)rand() / RAND_MAX;
    }
}

int verify(const float *host_in, const float *dev_out, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        if (fabs(host_in[N - 1 - i] - dev_out[i]) > 1e-5f) {
            printf("Mismatch at index %zu: host %f vs dev %f\n", i,
                   host_in[N - 1 - i], dev_out[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    const int N = 1 << 26; // ~67 million elements
    const size_t bytes = N * sizeof(float);
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Array size: %d elements (%.2f MB)\n", N, bytes / (1024.0 * 1024.0));

    // Allocate host memory
    float *h_in = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    if (!h_in || !h_ref) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }
    srand((unsigned int)time(NULL));
    generate_random_array(h_in, N);

    // Create reference result
    for (int i = 0; i < N; ++i) {
        h_ref[i] = h_in[N - 1 - i];
    }

    // Allocate device memory
    float *d_in, *d_out_naive, *d_out_shared;
    CUDA_CHECK(cudaMalloc((void **)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out_naive, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out_shared, bytes));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Run naive kernel
    int runs = 10;
    float time_naive = 0.0f;
    for (int r = 0; r < runs; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        reverse_naive<<<blocks, threadsPerBlock>>>(d_in, d_out_naive, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        time_naive += ms;
    }
    time_naive /= runs;
    printf("Naive kernel average time: %.3f ms\n", time_naive);

    // Run shared memory kernel
    float time_shared = 0.0f;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    for (int r = 0; r < runs; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        reverse_shared<<<blocks, threadsPerBlock, sharedMemSize>>>(d_in, d_out_shared, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        time_shared += ms;
    }
    time_shared /= runs;
    printf("Shared memory kernel average time: %.3f ms\n", time_shared);

    // Verify results
    float *h_out_naive = (float *)malloc(bytes);
    float *h_out_shared = (float *)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_out_naive, d_out_naive, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_shared, d_out_shared, bytes, cudaMemcpyDeviceToHost));

    printf("Verifying naive kernel result... ");
    if (verify(h_ref, h_out_naive, N))
        printf("PASSED\n");
    else
        printf("FAILED\n");

    printf("Verifying shared memory kernel result... ");
    if (verify(h_ref, h_out_shared, N))
        printf("PASSED\n");
    else
        printf("FAILED\n");

    // Clean up
    free(h_in);
    free(h_ref);
    free(h_out_naive);
    free(h_out_shared);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out_naive));
    CUDA_CHECK(cudaFree(d_out_shared));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```