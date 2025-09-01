```cuda
/*
Aim of the program:
Test your generalized kernel with several large N values (e.g., 10000, 500000, 2000000).

Thinking and design notes:
- The kernel we want to test is a simple elementwise vector addition: C[i] = A[i] + B[i].
  This is a good generic operation to verify correctness, performance and scalability
  across a range of problem sizes.
- We will use a templated kernel (float type for simplicity) and launch it for each
  value in the list {10000, 500000, 2000000}.  The code will allocate device memory,
  initialize host arrays, copy data to the device, run the kernel, copy the result back,
  verify correctness, and time each run with cudaEvent_t.
- The launch configuration uses 256 threads per block, which is a common choice and
  ensures enough parallelism for the larger N values.
- Verification is performed by checking that each element of C equals the sum of
  A and B within a small epsilon.
- Timing is reported in milliseconds for each N.  Any errors in memory allocation
  or kernel launch are printed and the program exits with a non‑zero status.
- The code is written to be self‑contained in a single .cu file, ready to compile
  with nvcc (e.g., `nvcc -O2 vector_add.cu -o vector_add`).
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void runTest(int N)
{
    size_t size = N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed for N=%d\n", N);
        exit(EXIT_FAILURE);
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;           // arbitrary values
        h_B[i] = 2.0f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());  // check kernel launch error

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify results
    float eps = 1e-5f;
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C[i] - expected) > eps) {
            errors++;
            if (errors <= 10) { // print first few errors
                fprintf(stderr, "Mismatch at index %d: host %f, device %f\n",
                        i, h_C[i], expected);
            }
        }
    }
    if (errors == 0) {
        printf("N = %d: SUCCESS, time = %.3f ms\n", N, milliseconds);
    } else {
        printf("N = %d: FAILURE, %d mismatches found, time = %.3f ms\n",
               N, errors, milliseconds);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main(void)
{
    int testSizes[] = {10000, 500000, 2000000};
    int numTests = sizeof(testSizes) / sizeof(testSizes[0]);

    for (int i = 0; i < numTests; ++i) {
        runTest(testSizes[i]);
    }

    return 0;
}
```