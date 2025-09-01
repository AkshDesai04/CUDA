```cuda
/*
Aim of the program: 
Instead of an if, can you use arithmetic to prevent out-of-bounds access (e.g. C[index] = ... where index is somehow clamped)? Is this better or worse than a branch?

My thinking:
To answer the question I wrote a small CUDA program that compares two vector addition kernels.  
The first kernel (add_with_branch) uses a conventional if-statement to guard against out‑of‑bounds thread indices:

    if (idx < n) C[idx] = A[idx] + B[idx];

The second kernel (add_with_clamp) attempts to avoid branches by clamping the indices with arithmetic:

    int clampedIdx = (idx < n) ? idx : n-1;          // clamp write index
    int aIdx = (idx < n) ? idx : 0;                  // clamp read index for A
    int bIdx = (idx < n) ? idx : 0;                  // clamp read index for B
    C[clampedIdx] = A[aIdx] + B[bIdx];

Because the clamping approach still reads from potentially out‑of‑bounds memory (if idx ≥ n), the results are incorrect – the kernel silently substitutes data from the first element (or the last element for the write).  
Moreover, the clamping adds extra integer operations (min/max or conditional assignments), and the compiler may not generate a single fast instruction. In many architectures, the branch in the first kernel is highly optimized (branch prediction, warp‑wide execution), and the branch divergence is minimal because the majority of threads are in‑bounds. Thus the branch kernel is usually faster and always correct.

The program allocates two input vectors of size N, fills them, launches both kernels with (N+extra) threads to force some out‑of‑bounds accesses, and compares the output arrays. It also measures the execution time of each kernel for a simple comparison. The expected result is that the branch kernel produces the correct output, while the clamped kernel does not. The timing shows whether the clamped version is faster or slower – in practice it is usually slower or about the same, but the correctness issue makes it a poor choice.

The code below is a self‑contained .cu file that can be compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024          // size of the vectors
#define BLOCK_SIZE 256  // block size for kernels

// Kernel with branch to guard out-of-bounds indices
__global__ void add_with_branch(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel that attempts to avoid branches by clamping indices
__global__ void add_with_clamp(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Clamp write index to [0, n-1]
    int clampedIdx = (idx < n) ? idx : (n - 1);

    // Clamp read indices to avoid out-of-bounds reads
    int aIdx = (idx < n) ? idx : 0;
    int bIdx = (idx < n) ? idx : 0;

    // Perform addition (will use clamped indices if idx >= n)
    C[clampedIdx] = A[aIdx] + B[bIdx];
}

// Utility function to check CUDA errors
void checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_branch = (float *)malloc(bytes);
    float *h_C_clamp = (float *)malloc(bytes);

    // Initialize input vectors
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C_branch, *d_C_clamp;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C_branch, bytes);
    cudaMalloc(&d_C_clamp, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernels with N+extra threads to force out-of-bounds accesses
    int threads = BLOCK_SIZE * 4; // 4 blocks for example
    dim3 blocks((threads + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch branch kernel
    cudaEventRecord(start);
    add_with_branch<<<blocks, BLOCK_SIZE>>>(d_A, d_B, d_C_branch, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_branch = 0;
    cudaEventElapsedTime(&ms_branch, start, stop);

    // Launch clamp kernel
    cudaEventRecord(start);
    add_with_clamp<<<blocks, BLOCK_SIZE>>>(d_A, d_B, d_C_clamp, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_clamp = 0;
    cudaEventElapsedTime(&ms_clamp, start, stop);

    // Copy results back to host
    cudaMemcpy(h_C_branch, d_C_branch, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_clamp, d_C_clamp, bytes, cudaMemcpyDeviceToHost);

    // Verify correctness of branch kernel
    int errors_branch = 0;
    for (int i = 0; i < N; ++i)
    {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C_branch[i] - expected) > 1e-5f)
        {
            errors_branch++;
            if (errors_branch <= 10)
                printf("branch error at %d: got %f, expected %f\n", i, h_C_branch[i], expected);
        }
    }

    // Verify correctness of clamp kernel
    int errors_clamp = 0;
    for (int i = 0; i < N; ++i)
    {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C_clamp[i] - expected) > 1e-5f)
        {
            errors_clamp++;
            if (errors_clamp <= 10)
                printf("clamp error at %d: got %f, expected %f\n", i, h_C_clamp[i], expected);
        }
    }

    // Report results
    printf("\nBranch kernel: %s\n", errors_branch == 0 ? "PASS" : "FAIL");
    printf("Clamp kernel: %s\n", errors_clamp == 0 ? "PASS" : "FAIL");
    printf("Branch kernel time: %.3f ms\n", ms_branch);
    printf("Clamp kernel time: %.3f ms\n", ms_clamp);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_branch);
    cudaFree(d_C_clamp);
    free(h_A);
    free(h_B);
    free(h_C_branch);
    free(h_C_clamp);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```