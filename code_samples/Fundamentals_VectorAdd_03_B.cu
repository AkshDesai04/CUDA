/*
Aim of the program: Test with `N = 255`, `N = 256`, and `N = 257` with a block size of 256. Verify correctness for all cases.

Thinking:
- The objective is to exercise CUDA kernel launch mechanics with varying problem sizes that are just below, equal to, and just above the block size.
- We will implement a simple element‑wise vector addition kernel: C[i] = A[i] + B[i].
- For each N in {255, 256, 257} we:
  1. Allocate host arrays A, B, C of size N and initialize A[i] = i, B[i] = 2*i.
  2. Allocate corresponding device arrays dA, dB, dC.
  3. Copy A and B to the device.
  4. Launch the kernel with blockDim.x = 256 and gridDim.x = ceil(N / 256).
  5. Copy the result back to host and verify that C[i] equals A[i] + B[i] for every element.
- The program will print a pass/fail message for each test case.
- CUDA error checking is incorporated via a macro to catch any failures in API calls or kernel launches.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check CUDA API calls for errors */
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Simple vector addition kernel */
__global__ void vectorAdd(const int *A, const int *B, int *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const int blockSize = 256;
    const int testNs[] = {255, 256, 257};
    const int numTests = sizeof(testNs) / sizeof(testNs[0]);

    for (int t = 0; t < numTests; ++t)
    {
        int N = testNs[t];
        printf("=== Test with N = %d, block size = %d ===\n", N, blockSize);

        /* Host allocations */
        int *h_A = (int*)malloc(N * sizeof(int));
        int *h_B = (int*)malloc(N * sizeof(int));
        int *h_C = (int*)malloc(N * sizeof(int));
        if (!h_A || !h_B || !h_C)
        {
            fprintf(stderr, "Failed to allocate host memory.\n");
            exit(EXIT_FAILURE);
        }

        /* Initialize host data */
        for (int i = 0; i < N; ++i)
        {
            h_A[i] = i;
            h_B[i] = 2 * i;
        }

        /* Device allocations */
        int *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc((void**)&d_A, N * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void**)&d_B, N * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void**)&d_C, N * sizeof(int)));

        /* Copy inputs to device */
        CHECK_CUDA(cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice));

        /* Kernel launch parameters */
        dim3 blockDim(blockSize);
        dim3 gridDim((N + blockSize - 1) / blockSize);

        /* Launch kernel */
        vectorAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaGetLastError());           // Check for launch errors
        CHECK_CUDA(cudaDeviceSynchronize());      // Wait for completion

        /* Copy result back to host */
        CHECK_CUDA(cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost));

        /* Verify results */
        int errors = 0;
        for (int i = 0; i < N; ++i)
        {
            int expected = h_A[i] + h_B[i];
            if (h_C[i] != expected)
            {
                if (errors < 10) // Print first few errors
                {
                    fprintf(stderr, "Mismatch at index %d: host=%d, device=%d, expected=%d\n",
                            i, h_C[i], expected, expected);
                }
                ++errors;
            }
        }

        if (errors == 0)
        {
            printf("Test passed: All %d elements match expected results.\n", N);
        }
        else
        {
            printf("Test failed: %d mismatches found.\n", errors);
        }

        /* Clean up */
        free(h_A);
        free(h_B);
        free(h_C);
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));

        printf("\n");
    }

    /* Reset device */
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
