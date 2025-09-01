```cuda
/*
Implement the operation out-of-place: `Z[i] = a * X[i] + Y[i]`

Thinking:
- The goal is to perform an element-wise linear combination of two input arrays X and Y
  and store the result into a third array Z without modifying X or Y (out-of-place).
- This is a classic vector operation, often referred to as "SAXPY" (Single-precision
  A·X Plus Y).  A CUDA implementation is straightforward: one thread per element
  computes `Z[i] = a * X[i] + Y[i]`.
- We will define a simple CUDA kernel `saxpyKernel` that takes pointers to the
  device arrays, the scalar a, and the number of elements n.  The kernel uses
  block indexing to map each thread to a unique index `i`.  If `i < n` the
  computation is performed.
- In the host code we:
  1. Allocate and initialize host arrays X and Y with some test values
     (for example, sin(i) and cos(i) or a simple pattern).
  2. Allocate device memory for X, Y, and Z.
  3. Copy X and Y to the device.
  4. Launch the kernel with an appropriate grid and block size.
  5. Copy the resulting Z array back to the host.
  6. (Optional) Verify a few elements and print them.
  7. Clean up all allocated memory.
- Error checking is performed after each CUDA API call using a simple macro
  `CHECK_CUDA` that prints the error string and exits on failure.
- The program is self‑contained and can be compiled with `nvcc`:
    nvcc -o saxpy saxpy.cu
  Running the executable will perform the operation and print the first 10
  results to demonstrate correctness.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",   \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),       \
                    cudaGetErrorString(err));                            \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

/* CUDA kernel: Z[i] = a * X[i] + Y[i] */
__global__ void saxpyKernel(const float *X, const float *Y, float *Z,
                            float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Z[idx] = a * X[idx] + Y[idx];
    }
}

/* Host function to perform SAXPY using CUDA */
int main(void) {
    /* Parameters */
    const int N = 1 << 20;          // Number of elements (1M)
    const float a = 2.5f;           // Scalar multiplier
    const size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_X = (float*)malloc(bytes);
    float *h_Y = (float*)malloc(bytes);
    float *h_Z = (float*)malloc(bytes);
    if (!h_X || !h_Y || !h_Z) {
        fprintf(stderr, "Failed to allocate host arrays\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_X[i] = sinf((float)i);   // Example pattern
        h_Y[i] = cosf((float)i);
    }

    /* Allocate device memory */
    float *d_X = NULL, *d_Y = NULL, *d_Z = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_X, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_Y, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_Z, bytes));

    /* Copy input data from host to device */
    CHECK_CUDA(cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Y, h_Y, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    saxpyKernel<<<gridSize, blockSize>>>(d_X, d_Y, d_Z, a, N);
    /* Check for launch errors */
    CHECK_CUDA(cudaGetLastError());
    /* Wait for GPU to finish */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_Z, d_Z, bytes, cudaMemcpyDeviceToHost));

    /* Verify and print first 10 results */
    printf("First 10 results of Z[i] = a * X[i] + Y[i] (a = %f):\n", a);
    for (int i = 0; i < 10; ++i) {
        printf("Z[%d] = %f\n", i, h_Z[i]);
    }

    /* Free device memory */
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
    CHECK_CUDA(cudaFree(d_Z));

    /* Free host memory */
    free(h_X);
    free(h_Y);
    free(h_Z);

    /* Reset device */
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```