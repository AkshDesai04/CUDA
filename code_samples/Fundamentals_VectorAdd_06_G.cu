/*
Verify the results of SAXPY on the CPU.

Thinking:
- SAXPY (Single-Precision A*X Plus Y) computes y = a * x + y.
- We'll use a scalar alpha, vector X (b), and vector Y (a) where a initially holds Y and b holds X.
- The program will allocate host arrays, initialize them, compute SAXPY on the CPU, then compute the same on the GPU using a CUDA kernel.
- After copying GPU results back to host, we'll compare each element with the CPU result within a tolerance (epsilon) to verify correctness.
- We'll output whether the verification passed or failed, and if any discrepancies are found, print the first few mismatched indices.
- The program uses basic CUDA error checking and includes timing of GPU execution for demonstration.
- All necessary includes and helper functions are provided. The code is self-contained in a single .cu file as requested.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N (1<<20)            // Vector size (1M elements)
#define BLOCK_SIZE 256       // Threads per block
#define EPSILON 1e-5f

// CUDA error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CPU implementation of SAXPY: y = alpha * x + y
void cpu_saxpy(float *y, const float *x, float alpha, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

// CUDA kernel for SAXPY
__global__ void saxpy_kernel(float *y, const float *x, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

// Helper to fill array with random floats
void fill_random(float *arr, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i] = (float)rand() / RAND_MAX * 10.0f; // values in [0,10)
    }
}

int main(void) {
    // Seed RNG
    srand((unsigned)time(NULL));

    // Host arrays
    float *h_y_cpu = (float *)malloc(N * sizeof(float));
    float *h_x     = (float *)malloc(N * sizeof(float));
    float *h_y_gpu = (float *)malloc(N * sizeof(float));

    if (!h_y_cpu || !h_x || !h_y_gpu) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize vectors
    fill_random(h_y_cpu, N); // This will be Y
    fill_random(h_x,     N); // X

    // Copy Y to GPU array
    float *d_y, *d_x;
    CHECK_CUDA(cudaMalloc((void **)&d_y, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_y, h_y_cpu, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x,     N * sizeof(float), cudaMemcpyHostToDevice));

    // CPU SAXPY
    const float alpha = 2.5f;
    cpu_saxpy(h_y_cpu, h_x, alpha, N);

    // GPU SAXPY
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    saxpy_kernel<<<grid_size, BLOCK_SIZE>>>(d_y, d_x, alpha, N);
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch errors

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    printf("GPU SAXPY executed in %.3f ms.\n", elapsed_ms);

    // Copy GPU result back
    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        float diff = h_y_cpu[i] - h_y_gpu[i];
        if (diff < 0) diff = -diff;
        if (diff > EPSILON) {
            mismatches++;
            if (mismatches <= 10) {
                printf("Mismatch at index %d: CPU=%f GPU=%f diff=%f\n",
                       i, h_y_cpu[i], h_y_gpu[i], diff);
            }
        }
    }

    if (mismatches == 0) {
        printf("Verification PASSED: GPU results match CPU results.\n");
    } else {
        printf("Verification FAILED: %d mismatches found.\n", mismatches);
    }

    // Clean up
    free(h_y_cpu);
    free(h_x);
    free(h_y_gpu);
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
