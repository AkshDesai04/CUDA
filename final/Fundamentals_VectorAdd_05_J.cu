/*
 Verify the results of multiplication on the CPU.

 The goal of this CUDA program is to demonstrate how to perform element‑wise multiplication
 of two large vectors on the GPU and then verify that the CPU produces the same results.
 The program will:
 1. Allocate and initialize two host vectors A and B with random floating‑point values.
 2. Copy these vectors to the device.
 3. Launch a simple kernel that computes C_g[i] = A[i] * B[i] for every element.
 4. Copy the GPU result back to the host into C_g.
 5. Compute the same multiplication on the CPU, storing the result in C_c.
 6. Compare C_g and C_c element‑wise and report whether the verification passed.
 
 The program uses a macro for error checking, and it reports both CPU and GPU timings
 for reference. The data size is chosen to be large enough (1 million elements) to
 make the GPU computation meaningful, yet small enough to run quickly on typical
 development machines.

 The reasoning behind this approach is straightforward:
 * GPU kernels are highly parallel and efficient for data‑parallel tasks like vector
   multiplication.
 * Performing a second, independent CPU calculation allows us to confirm the correctness
   of the GPU implementation without relying on external test harnesses.
 * A simple element‑wise product is easy to understand and verify, yet the code
   structure (memory allocation, kernel launch, data transfer) is generic enough
   to be extended to more complex operations such as matrix multiplication.

 Note that for large vectors the floating‑point results may differ slightly due
 to the different ordering of operations on the CPU and GPU. We therefore use a
 tolerance in the comparison step.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N (1 << 20)          // 1,048,576 elements
#define BLOCK_SIZE 256
#define TOLERANCE 1e-5f

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// GPU kernel for element-wise multiplication
__global__ void vectorMul(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

int main(void) {
    // Allocate host memory
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C_g = (float *)malloc(N * sizeof(float)); // GPU result
    float *h_C_c = (float *)malloc(N * sizeof(float)); // CPU result

    if (!h_A || !h_B || !h_C_g || !h_C_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Seed random number generator
    srand((unsigned)time(NULL));

    // Initialize host vectors with random values
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX; // [0,1)
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorMul<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_g, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU multiplication
    for (int i = 0; i < N; ++i) {
        h_C_c[i] = h_A[i] * h_B[i];
    }

    // Verify results
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        float diff = fabsf(h_C_g[i] - h_C_c[i]);
        if (diff > TOLERANCE) {
            mismatches++;
            if (mismatches <= 10) { // Print first few mismatches
                printf("Mismatch at index %d: GPU=%f, CPU=%f, diff=%f\n",
                       i, h_C_g[i], h_C_c[i], diff);
            }
        }
    }

    if (mismatches == 0) {
        printf("Verification passed: All elements match within tolerance.\n");
    } else {
        printf("Verification failed: %d mismatches found.\n", mismatches);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_g);
    free(h_C_c);

    return 0;
}
