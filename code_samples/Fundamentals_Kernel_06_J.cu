```cuda
/*
 * Aim: Add robust error checking around every CUDA API call (`cudaMalloc`, `cudaMemcpy`, `cudaFree`).
 *
 * Thinking:
 * 1. Create a macro `CUDA_CHECK(call)` that executes a CUDA API call, captures the returned
 *    `cudaError_t`, and if it is not `cudaSuccess`, prints an informative error message
 *    (including file name and line number) and exits the program. This ensures every API call
 *    is verified for success.
 * 2. Use this macro around all `cudaMalloc`, `cudaMemcpy`, and `cudaFree` calls.
 * 3. Additionally, after a kernel launch, invoke `cudaGetLastError()` and
 *    `cudaDeviceSynchronize()` through the same macro to catch launch and execution errors.
 * 4. Demonstrate the approach with a simple vector addition kernel.
 * 5. Keep the code self‑contained and free of extraneous dependencies.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for robust error checking of CUDA API calls */
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s (%d)\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);  \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* Simple vector addition kernel */
__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main(void) {
    const int N = 1 << 20;          // 1M elements
    const size_t size = N * sizeof(float);

    /* Allocate and initialize host memory */
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory with robust error checking */
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    /* Copy host data to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());          // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize());     // Wait for completion and catch execution errors

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* Verify results */
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            if (errors < 10) { // Report first few errors
                fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n",
                        i, h_c[i], expected);
            }
            ++errors;
        }
    }
    if (errors == 0) {
        printf("Vector addition successful, no errors detected.\n");
    } else {
        printf("Vector addition completed with %d errors.\n", errors);
    }

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    /* Free host memory */
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```