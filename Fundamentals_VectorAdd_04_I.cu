/*
Aim: Time the subtraction kernel. Is it significantly different from addition?

Thinking:
To answer whether the subtraction kernel is significantly different from addition, we implement both kernels in CUDA,
measure their execution times using cudaEvent, and compare the results. 
Both kernels perform element‑wise operations on large vectors to ensure that kernel launch overhead is negligible
compared to the actual arithmetic cost. 
We will:
1. Allocate host and device memory for two input vectors and an output vector.
2. Initialize the input vectors with random values.
3. Run the addition kernel, time it, and copy back the result.
4. Run the subtraction kernel, time it, and copy back the result.
5. Print the measured times and the ratio between them.
6. Optionally verify that the outputs are as expected.

Since both addition and subtraction are simple element‑wise arithmetic operations, they should have very similar
execution times; any difference is likely due to compiler optimizations or instruction scheduling rather than
fundamental algorithmic differences. The program will demonstrate this by printing the timings.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N (1 << 20)          // 1M elements
#define BLOCK_SIZE 256

// CUDA kernel for element-wise addition
__global__ void addKernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// CUDA kernel for element-wise subtraction
__global__ void subKernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] - b[idx];
}

// Helper to check CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c_add = (float*)malloc(N * sizeof(float));
    float *h_c_sub = (float*)malloc(N * sizeof(float));

    // Initialize host data
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_a[i] = rand() / (float)RAND_MAX * 100.0f;
        h_b[i] = rand() / (float)RAND_MAX * 100.0f;
    }

    // Device pointers
    float *d_a, *d_b, *d_c;
    checkCudaError(cudaMalloc((void**)&d_a, N * sizeof(float)), "cudaMalloc d_a");
    checkCudaError(cudaMalloc((void**)&d_b, N * sizeof(float)), "cudaMalloc d_b");
    checkCudaError(cudaMalloc((void**)&d_c, N * sizeof(float)), "cudaMalloc d_c");

    // Copy inputs to device
    checkCudaError(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy h_a->d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy h_b->d_b");

    // Determine grid dimensions
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Events for timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "EventCreate start");
    checkCudaError(cudaEventCreate(&stop), "EventCreate stop");

    // ---------------- Addition ----------------
    checkCudaError(cudaEventRecord(start, 0), "EventRecord start add");
    addKernel<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    checkCudaError(cudaEventRecord(stop, 0), "EventRecord stop add");
    checkCudaError(cudaEventSynchronize(stop), "EventSynchronize stop add");
    float timeAdd = 0.0f;
    checkCudaError(cudaEventElapsedTime(&timeAdd, start, stop), "EventElapsedTime add");

    // Copy result back
    checkCudaError(cudaMemcpy(h_c_add, d_c, N * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy d_c->h_c_add");

    // ---------------- Subtraction ----------------
    checkCudaError(cudaEventRecord(start, 0), "EventRecord start sub");
    subKernel<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    checkCudaError(cudaEventRecord(stop, 0), "EventRecord stop sub");
    checkCudaError(cudaEventSynchronize(stop), "EventSynchronize stop sub");
    float timeSub = 0.0f;
    checkCudaError(cudaEventElapsedTime(&timeSub, start, stop), "EventElapsedTime sub");

    // Copy result back
    checkCudaError(cudaMemcpy(h_c_sub, d_c, N * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy d_c->h_c_sub");

    // Print timings
    printf("Addition kernel time:   %f ms\n", timeAdd);
    printf("Subtraction kernel time: %f ms\n", timeSub);
    printf("Time ratio (sub/add): %f\n", timeSub / timeAdd);

    // Simple verification (optional)
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c_add[i] - (h_a[i] + h_b[i])) > 1e-5f ||
            fabs(h_c_sub[i] - (h_a[i] - h_b[i])) > 1e-5f) {
            mismatches++;
            if (mismatches < 5) {
                printf("Mismatch at %d: add %f vs %f, sub %f vs %f\n",
                       i, h_c_add[i], h_a[i] + h_b[i],
                       h_c_sub[i], h_a[i] - h_b[i]);
            }
        }
    }
    if (mismatches == 0) {
        printf("Verification passed: all results are correct.\n");
    } else {
        printf("Verification failed: %d mismatches.\n", mismatches);
    }

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c_add); free(h_c_sub);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
