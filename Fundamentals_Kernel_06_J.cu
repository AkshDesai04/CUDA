/*
Add robust error checking around every CUDA API call (cudaMalloc, cudaMemcpy, cudaFree).

Thinking:
- The goal is to demonstrate how to wrap each CUDA API call with error checking.
- We'll create a simple vector addition kernel and perform all host and device operations with
  error checking.
- A macro `CUDA_CHECK` will be defined that takes a CUDA function call, checks its return
  value, prints a detailed error message (including the file and line number), and exits
  if the call fails.
- For kernel launches, we'll check `cudaGetLastError()` immediately after launch and
  also check `cudaDeviceSynchronize()` to catch errors that occur during execution.
- We'll also wrap `cudaFree` calls with the same macro to ensure resources are freed
  properly.
- All calls (`cudaMalloc`, `cudaMemcpy`, `cudaFree`, kernel launch, and
  `cudaDeviceSynchronize`) will be protected by this macro.
- The program will allocate two input arrays on the host, copy them to device memory,
  launch the kernel to add them, copy the result back to host, verify a few results,
  and then clean up.
- The program uses standard C (`#include <stdio.h>` and `#include <stdlib.h>`) and
  includes `<cuda_runtime.h>` for CUDA runtime API functions.
- Error messages will include the exact CUDA error string from `cudaGetErrorString`
  for clarity.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/* Simple vector addition kernel */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20; /* 1M elements */
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    /* Copy host data to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Wait for device to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Verify results for a few elements */
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n",
                    i, h_C[i], expected);
            errors++;
        }
    }
    if (errors == 0) {
        printf("Result verified successfully.\n");
    } else {
        printf("Found %d errors in verification.\n", errors);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
