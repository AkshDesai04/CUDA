```c
/*
Aim of the program: Implement `C[i] = (A[i] > threshold) ? 1 : 0` (thresholding).

Thinking:
1. The task is to perform a simple element‑wise threshold operation on an array `A`.
   For each element we need to compare it against a scalar `threshold` and write
   either 1 or 0 into the corresponding element of array `C`.  
2. Since the operation is embarrassingly parallel, a CUDA kernel is the natural
   fit: each thread handles one index `i`.  
3. We will write a kernel `thresholdKernel` that receives a pointer to `A`,
   a pointer to the output `C`, the threshold value, and the size of the
   array.  The kernel will compute the global index `idx` and, if it is
   within bounds, perform the comparison and store the result.  
4. In `main()` we will:
   - Allocate and initialize a host array `h_A` with some example data.
   - Allocate device memory for `d_A` and `d_C`.
   - Copy `h_A` to `d_A`.
   - Launch the kernel with a suitable grid/block configuration.
   - Copy the result `d_C` back to host `h_C`.
   - Print a few results to verify correctness.
   - Free all allocated memory and reset the device.
5. For robustness, we will include a simple CUDA error‑checking macro
   `CUDA_CHECK` that aborts if a CUDA call fails.
6. The code is self‑contained and can be compiled with `nvcc threshold.cu -o threshold`
   and executed on a system with an NVIDIA GPU.

Edge cases considered:
- The kernel guard `if (idx < N)` ensures no out‑of‑bounds writes for the last
  partial block.
- Using `float` for input values keeps the example simple, but the kernel
  could easily be templated for other types.
- We set the block size to 256 threads, a common choice, but this can be
  tuned for performance.

The program demonstrates the core pattern of:
   1) memory allocation on host/device,
   2) data transfer,
   3) kernel launch,
   4) result retrieval, and
   5) cleanup.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Simple error checking macro */
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",    \
                    __func__, __FILE__, __LINE__,                \
                    cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

/* CUDA kernel performing thresholding: C[i] = (A[i] > threshold) ? 1 : 0 */
__global__ void thresholdKernel(const float* __restrict__ A,
                                 int* __restrict__ C,
                                 float threshold,
                                 int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = (A[idx] > threshold) ? 1 : 0;
    }
}

int main(void)
{
    const int N = 1024;                    /* Number of elements */
    const float threshold = 0.5f;          /* Threshold value */

    /* Host memory allocation */
    float* h_A = (float*)malloc(N * sizeof(float));
    int*   h_C = (int*)  malloc(N * sizeof(int));
    if (!h_A || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host input array with some values */
    for (int i = 0; i < N; ++i) {
        h_A[i] = ((float)i / N);          /* values from 0.0 to just below 1.0 */
    }

    /* Device memory allocation */
    float* d_A = NULL;
    int*   d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(int)));

    /* Copy input data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch the threshold kernel */
    thresholdKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, threshold, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Print first 16 results for verification */
    printf("First 16 results (index: value):\n");
    for (int i = 0; i < 16 && i < N; ++i) {
        printf("%3d: %d (A=%.4f)\n", i, h_C[i], h_A[i]);
    }

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));

    /* Free host memory */
    free(h_A);
    free(h_C);

    /* Reset device */
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```