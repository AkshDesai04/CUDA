```cuda
/* 
Aim of the program (word for word from the prompt): 
Explain in a comment why calculating the number of blocks as `(N + BS - 1) / BS` is better than `N / BS`.

My thinking while writing this comment:
- The main goal of the program is to illustrate why the formula `(N + BS - 1) / BS` correctly calculates the number of thread blocks needed to cover `N` elements when each block handles `BS` threads.
- If we simply use integer division `N / BS`, any leftover elements (when `N` is not a multiple of `BS`) will be ignored because integer division truncates toward zero. This means the kernel would under‑process the data, leaving the last few elements uncomputed.
- By adding `BS - 1` before dividing, we effectively round up the division. This ensures that if there is any remainder, an extra block is launched to process those leftover elements.
- The expression `(N + BS - 1) / BS` is equivalent to `ceil(N / BS)` but uses only integer arithmetic, which is efficient and avoids floating‑point operations.
- In CUDA, launching fewer blocks than needed can lead to incorrect results, while launching too many blocks (though harmless) is wasteful. Thus, the round‑up formula provides the minimal safe number of blocks.
- This pattern is widely used in CUDA programming for grid size calculation.

The program below demonstrates this by performing a simple vector addition and prints the number of blocks computed by both formulas to illustrate the difference.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BS 256  // block size

// CUDA kernel to add two vectors
__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

// Utility macro for CUDA error checking
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.", \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main(void)
{
    const int N = 1000;  // number of elements (intentionally not a multiple of BS)
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Calculate grid size using both methods
    int gridSize_div = N / BS;                     // naive division
    int gridSize_ceil = (N + BS - 1) / BS;         // round‑up division

    printf("Using naive division (N/BS): %d blocks\n", gridSize_div);
    printf("Using round‑up formula ((N+BS-1)/BS): %d blocks\n", gridSize_ceil);

    // Launch kernel with round‑up formula (safer choice)
    vecAdd<<<gridSize_ceil, BS>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results for a few elements
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            if (errors < 10) {
                printf("Mismatch at index %d: got %f, expected %f\n",
                       i, h_C[i], expected);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("All results are correct.\n");
    } else {
        printf("Found %d mismatches.\n", errors);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```