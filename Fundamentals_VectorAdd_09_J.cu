/*
Implement `D[i] = sqrt(A[i]*A[i] + B[i]*B[i])` (vector magnitude for 2D vectors).

Thinking:
1. The goal is to compute the magnitude of 2D vectors where each vector component is stored in separate arrays A and B.
2. The operation is embarrassingly parallel: each element D[i] depends only on A[i] and B[i].
3. We will write a simple CUDA kernel that computes D[i] for each index i.
4. Host code will allocate memory for A, B, D, initialize A and B with sample data, copy to device, launch kernel, copy result back, and verify.
5. Use float for simplicity; the sqrtf function is available in CUDA.
6. Use typical block size of 256 threads and compute the number of blocks accordingly.
7. Include error checking for all CUDA API calls.
8. Measure time for kernel execution using cudaEvent to show performance.
9. The program is self-contained and can be compiled with nvcc: `nvcc -o vecmag vecmag.cu`
10. Output a few sample results for sanity check.

The code below implements all of the above. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Kernel to compute vector magnitudes
__global__ void vectorMagnitude(const float *A, const float *B, float *D, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = A[idx];
        float b = B[idx];
        D[idx] = sqrtf(a * a + b * b);
    }
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void)
{
    const int N = 1 << 20;           // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_D = (float*)malloc(size);

    if (!h_A || !h_B || !h_D) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with sample data
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i / N;          // values in [0,1)
        h_B[i] = (float)(N - i) / N;    // values in (0,1]
    }

    // Allocate device memory
    float *d_A, *d_B, *d_D;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_D, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch kernel and time it
    CUDA_CHECK(cudaEventRecord(start, 0));
    vectorMagnitude<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for kernel to finish
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Compute elapsed time
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %f ms\n", ms);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost));

    // Simple verification: print first 5 results
    printf("Sample results (index : A : B : D):\n");
    for (int i = 0; i < 5; ++i) {
        printf("%d : %f : %f : %f\n", i, h_A[i], h_B[i], h_D[i]);
    }

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_D));
    free(h_A);
    free(h_B);
    free(h_D);

    return 0;
}
