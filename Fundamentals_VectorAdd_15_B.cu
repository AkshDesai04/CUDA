```cuda
/*
 * Compare the performance of the fused kernel (`C=A+B`, `D=A-B`) against two separate kernel launches (one for addition, one for subtraction). The fused kernel should be faster.
 *
 * Thought process:
 * 1. Decide on the data type: float for simplicity.
 * 2. Choose array size N. A large enough size (e.g., 1<<24) to amortize launch overhead.
 * 3. Allocate three arrays on device: A, B, C, D.
 * 4. Initialize host arrays A_h and B_h with random floats, copy to device.
 * 5. Implement three kernels:
 *    - addKernel: C[i] = A[i] + B[i];
 *    - subKernel: D[i] = A[i] - B[i];
 *    - fusedKernel: C[i] = A[i] + B[i]; D[i] = A[i] - B[i];
 * 6. For timing, use cudaEvent_t start, stop and cudaEventElapsedTime which gives time in ms.
 * 7. Perform warmup runs to avoid cold-start effects.
 * 8. For fair comparison, run each kernel multiple times (e.g., 10) and average.
 * 9. Ensure we use same grid/block dimensions for all kernels.
 * 10. Print the average times and confirm fused kernel is faster.
 * 11. Clean up memory and events.
 * 12. Add error checking macro for CUDA API calls to catch issues.
 * 13. Provide instructions in comments for compiling with nvcc.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void addKernel(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void subKernel(const float *A, const float *B, float *D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        D[idx] = A[idx] - B[idx];
    }
}

__global__ void fusedKernel(const float *A, const float *B, float *C, float *D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = A[idx];
        float b = B[idx];
        C[idx] = a + b;
        D[idx] = a - b;
    }
}

int main(void) {
    const int N = 1 << 24;          // 16M elements
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    size_t bytes = N * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);

    // Initialize host arrays with random values
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // range [-1,1]
        h_B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    float *d_A, *d_B, *d_C, *d_D;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_D, bytes));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int warmup = 2;
    const int trials = 10;
    float time_add = 0.0f, time_sub = 0.0f, time_fused = 0.0f;

    // Warmup and timing for add + sub kernels
    for (int t = 0; t < warmup + trials; ++t) {
        CHECK_CUDA(cudaEventRecord(start, 0));

        addKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        subKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_D, N);

        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        if (t >= warmup) time_add += ms; // same ms for both kernels together
    }

    // Warmup and timing for fused kernel
    for (int t = 0; t < warmup + trials; ++t) {
        CHECK_CUDA(cudaEventRecord(start, 0));

        fusedKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N);

        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        if (t >= warmup) time_fused += ms;
    }

    time_add /= trials;
    time_fused /= trials;

    printf("Average time (add + sub): %.3f ms\n", time_add);
    printf("Average time (fused):    %.3f ms\n", time_fused);
    if (time_fused < time_add)
        printf("Fused kernel is faster by %.2f%%\n", (time_add - time_fused) / time_add * 100.0f);
    else
        printf("Fused kernel is NOT faster by %.2f%%\n", (time_fused - time_add) / time_add * 100.0f);

    // Cleanup
    free(h_A);
    free(h_B);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```