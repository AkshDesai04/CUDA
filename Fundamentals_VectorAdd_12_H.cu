/*
Aim: Verify the matrix addition result on the CPU.

Thinking:
The task is to write a CUDA program that performs matrix addition on the GPU and then verifies the result by recomputing the same addition on the CPU.  
To keep the example self‑contained and easy to compile, I chose a fixed square matrix size (1024×1024).  
The program flow is:

1. Allocate host arrays for matrices A, B, C_g (GPU result), and C_c (CPU result).  
2. Fill A and B with pseudo‑random floating‑point values.  
3. Allocate device arrays, copy A and B to device memory.  
4. Launch a simple element‑wise addition kernel.  
5. Copy the device result back to host array C_g.  
6. Compute the same addition on the CPU into C_c.  
7. Compare C_g and C_c element‑wise, using a small tolerance for floating‑point inaccuracies.  
8. Report success or failure, and optionally print timing information for both GPU and CPU paths.

I also wrapped CUDA API calls in a small error‑checking macro to make debugging easier.  The kernel uses a 2D grid of 16×16 thread blocks, which works well for the chosen matrix size.  All code is in a single .cu file and ready to compile with `nvcc`.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024                // Matrix dimension (N x N)
#define BLOCK_SIZE 16         // Block size for kernel
#define EPSILON 1e-5f         // Tolerance for floating point comparison

/* Macro to check CUDA API calls for errors */
#define CUDA_CHECK(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA error in file '%s' at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

/* Kernel for element-wise matrix addition */
__global__ void matrixAdd(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row < n && col < n) {
        int idx = row * n + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    int size = N * N;
    size_t bytes = size * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_g = (float *)malloc(bytes); // Result from GPU
    float *h_C_c = (float *)malloc(bytes); // Result from CPU

    if (!h_A || !h_B || !h_C_g || !h_C_c) {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        return EXIT_FAILURE;
    }

    /* Initialize matrices with random values */
    srand((unsigned int)time(NULL));
    for (int i = 0; i < size; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    /* Copy host matrices to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Define grid and block dimensions */
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* Timing GPU execution */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    /* Launch kernel */
    matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    /* Stop timing */
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C_g, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify result on CPU */
    clock_t cpu_start = clock();
    for (int i = 0; i < size; ++i) {
        h_C_c[i] = h_A[i] + h_B[i];
    }
    clock_t cpu_end = clock();
    double cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    /* Compare GPU and CPU results */
    int correct = 1;
    for (int i = 0; i < size; ++i) {
        float diff = fabsf(h_C_g[i] - h_C_c[i]);
        if (diff > EPSILON) {
            fprintf(stderr, "Mismatch at index %d: GPU=%f CPU=%f diff=%f\n",
                    i, h_C_g[i], h_C_c[i], diff);
            correct = 0;
            break;
        }
    }

    if (correct) {
        printf("Verification passed: GPU and CPU results match.\n");
    } else {
        printf("Verification failed: GPU and CPU results do not match.\n");
    }

    printf("GPU time: %.3f ms\n", ms);
    printf("CPU time: %.3f ms\n", cpu_time);

    /* Clean up */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_g);
    free(h_C_c);

    return EXIT_SUCCESS;
}
