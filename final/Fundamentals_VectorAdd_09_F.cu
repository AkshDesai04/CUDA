/*
Verify the results on the CPU.

Thought process:
- The goal is to confirm that computations performed on the GPU produce the same results as identical computations performed on the CPU.
- I chose a simple element‑wise vector addition as the workload because it is easy to understand, quick to compute, and allows direct comparison of each element.
- The program will:
  1. Allocate two input arrays `a` and `b` and an output array `c_gpu` on the host.
  2. Initialize `a` and `b` with known values (e.g., a[i] = i, b[i] = 2*i) so that the expected result is predictable.
  3. Allocate corresponding device memory and copy `a` and `b` to the device.
  4. Launch a CUDA kernel that performs element‑wise addition and writes the result to `c_gpu`.
  5. Copy the GPU result back to the host.
  6. Perform the same addition on the CPU into an array `c_cpu`.
  7. Compare `c_gpu` and `c_cpu` element by element. If any element differs beyond a small epsilon, report a mismatch; otherwise confirm that the verification passed.
- I also include helper macros for CUDA error checking to make debugging easier.
- The code is written in C and can be compiled with `nvcc program.cu -o program`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024            // Size of vectors
#define BLOCK_SIZE 256    // Threads per block
#define EPSILON 1e-5f     // Tolerance for floating point comparison

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",            \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    float *h_a = NULL;
    float *h_b = NULL;
    float *h_c_gpu = NULL;
    float *h_c_cpu = NULL;
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    h_a = (float *)malloc(bytes);
    h_b = (float *)malloc(bytes);
    h_c_gpu = (float *)malloc(bytes);
    h_c_cpu = (float *)malloc(bytes);

    if (h_a == NULL || h_b == NULL || h_c_gpu == NULL || h_c_cpu == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with deterministic values
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(2 * i);
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    // Compute result on CPU
    for (int i = 0; i < N; i++) {
        h_c_cpu[i] = h_a[i] + h_b[i];
    }

    // Verify results
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_c_gpu[i] - h_c_cpu[i]);
        if (diff > EPSILON) {
            if (mismatches < 10) { // Print first few mismatches
                printf("Mismatch at index %d: GPU=%f, CPU=%f, diff=%f\n",
                       i, h_c_gpu[i], h_c_cpu[i], diff);
            }
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("Verification passed: GPU and CPU results match within tolerance.\n");
    } else {
        printf("Verification failed: %d mismatches found.\n", mismatches);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);

    return EXIT_SUCCESS;
}
