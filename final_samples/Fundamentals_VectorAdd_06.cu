```cuda
/*
Implement a kernel for the SAXPY (Single-precision A*X Plus Y) operation. The operation is `Y = a*X + Y`, where `a` is a scalar float, and `X` and `Y` are vectors of floats. The scalar `a` should be passed as an argument to the kernel.

Thinking:
1. The SAXPY operation updates each element of Y in place: Y[i] = a * X[i] + Y[i].
2. A CUDA kernel can perform this in parallel: each thread handles one element.
3. The kernel signature will include the vector length `N`, the scalar `a`, a pointer to input vector `X`, and a pointer to output/in‑place vector `Y`.
4. Thread indexing: `int idx = blockIdx.x * blockDim.x + threadIdx.x;`. Each thread will check `idx < N` before performing the operation.
5. Grid and block configuration: a common block size of 256 threads is used; the grid size is computed to cover all `N` elements.
6. Host code will:
   - Allocate and initialize host vectors X and Y.
   - Allocate device memory for X and Y.
   - Copy data from host to device.
   - Launch the kernel.
   - Copy the result Y back to host.
   - Optionally print a few results to verify correctness.
   - Clean up memory.
7. Include simple error checking macro to validate CUDA API calls.
8. The program is self‑contained, compiling with `nvcc`, and prints the first few elements of the result vector.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple CUDA error checking macro */
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__,                \
                    cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

/* SAXPY kernel: Y = a*X + Y */
__global__ void saxpy_kernel(int N, float a, const float *X, float *Y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Y[idx] = a * X[idx] + Y[idx];
    }
}

int main(void)
{
    /* Problem size */
    const int N = 1 << 20;   /* 1M elements */
    const float a = 2.0f;    /* scalar multiplier */

    /* Host memory allocation */
    float *h_X = (float *)malloc(N * sizeof(float));
    float *h_Y = (float *)malloc(N * sizeof(float));
    if (!h_X || !h_Y) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_X[i] = (float)i;
        h_Y[i] = 0.5f * (float)i;
    }

    /* Device memory allocation */
    float *d_X = NULL;
    float *d_Y = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_X, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_Y, N * sizeof(float)));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Y, h_Y, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch SAXPY kernel */
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, a, d_X, d_Y);

    /* Check for kernel launch errors */
    CHECK_CUDA(cudaGetLastError());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Verify and print first 10 results */
    printf("First 10 results of Y = a*X + Y:\n");
    for (int i = 0; i < 10; ++i) {
        printf("Y[%d] = %f\n", i, h_Y[i]);
    }

    /* Clean up */
    free(h_X);
    free(h_Y);
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));

    /* Reset device (optional) */
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```