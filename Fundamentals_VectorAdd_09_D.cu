```cuda
/*
Aim of the program: Handle large vectors with boundary checks.

Thinking:
1. The requirement is to process large vectors safely, so we need to use CUDA
   kernels that guard against out-of-bounds memory accesses.  
2. A simple and common operation that demonstrates boundary checking is
   vector addition.  The kernel will compute `C[i] = A[i] + B[i]` only if
   `i < n`.  
3. For large vectors (potentially hundreds of millions of elements) we must:
   - Use a large grid of blocks so that every element is processed.
   - Compute the global index (`i`) as `blockIdx.x * blockDim.x + threadIdx.x`.
   - Include a conditional `if (i < n)` inside the kernel to prevent
     threads whose indices exceed the vector size from accessing memory.
4. In the host code we:
   - Parse an optional command line argument to set the vector length `n`.
   - Allocate pinned host memory for speed and ease of error checking.
   - Allocate device memory with `cudaMalloc` and handle allocation errors.
   - Initialize host input vectors with deterministic data.
   - Copy inputs to device with `cudaMemcpy`.
   - Launch the kernel with a block size of 256 threads and enough blocks to
     cover all elements.
   - Use `cudaDeviceSynchronize` and check the return code for any launch
     errors.
   - Copy the result back to the host and verify a few entries.
   - Clean up all allocated memory.
   - Measure the kernel launch time using CUDA events for completeness.
5. The program is written in plain C/C++ with CUDA extensions and is a single
   `.cu` file that can be compiled with `nvcc`.
6. All error codes are handled gracefully and informative messages are
   printed to `stderr`.
7. No external libraries are required; everything is part of the CUDA
   runtime.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// Kernel: vector addition with boundary check
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

// Simple error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s at line %d: %s (%d)\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);     \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Helper: generate data on host
void initHostData(float *A, float *B, int n)
{
    for (int i = 0; i < n; ++i) {
        A[i] = (float)i;
        B[i] = (float)(n - i);
    }
}

int main(int argc, char *argv[])
{
    // Default vector size: 1e8 elements (~400 MB per array)
    size_t n = 100000000;
    if (argc > 1) {
        n = (size_t)strtoull(argv[1], NULL, 10);
        if (n == 0) {
            fprintf(stderr, "Invalid vector size: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
    }

    size_t bytes = n * sizeof(float);
    printf("Vector size: %zu elements (%zu MB)\n", n, bytes / (1024 * 1024));

    // Allocate host memory (pinned for better transfer performance)
    float *h_A = NULL, *h_B = NULL, *h_C = NULL;
    CUDA_CHECK(cudaMallocHost((void**)&h_A, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_B, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_C, bytes));

    // Initialize host data
    initHostData(h_A, h_B, n);

    // Allocate device memory
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Determine kernel launch configuration
    const int threadsPerBlock = 256;
    int blocksPerGrid = (int)((n + threadsPerBlock - 1) / threadsPerBlock);

    printf("Launching kernel with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, (int)n);

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %f ms\n", ms);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print first 5 and last 5 results
    printf("Result verification (first 5 elements):\n");
    for (int i = 0; i < 5; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }
    printf("Result verification (last 5 elements):\n");
    for (int i = (int)n - 5; i < (int)n; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    printf("Program completed successfully.\n");
    return EXIT_SUCCESS;
}
```