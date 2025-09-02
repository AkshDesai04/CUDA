```cpp
/* 
Wrap all three `cudaMemcpy` calls (HtoD for A, HtoD for B, DtoH for C). 
In this program I aim to demonstrate a simple vector addition on the GPU, 
ensuring that all three memory transfer operations between host and device 
are correctly wrapped using `cudaMemcpy`. 

My thinking process:
1. I will define a small kernel `addVectors` that performs element‑wise addition.
2. In `main`, I allocate three host arrays (`h_a`, `h_b`, `h_c`) of length `N`.
3. I initialize `h_a` and `h_b` with some values.
4. I allocate corresponding device arrays (`d_a`, `d_b`, `d_c`) on the GPU.
5. I wrap the first `cudaMemcpy` as Host‑to‑Device for array `A` (`h_a` → `d_a`).
6. I wrap the second `cudaMemcpy` as Host‑to‑Device for array `B` (`h_b` → `d_b`).
7. I launch the kernel to compute `d_c = d_a + d_b`.
8. I wrap the third `cudaMemcpy` as Device‑to‑Host for array `C` (`d_c` → `h_c`).
9. Finally I print a few results, free all allocated memory, and exit. 
10. I also include a simple macro `checkCudaError` for robust error checking after each CUDA call.

This ensures that all three `cudaMemcpy` operations are explicitly shown and wrapped in the program.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024  // Size of vectors

/* Kernel to add two vectors element-wise */
__global__ void addVectors(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/* Helper macro for error checking */
#define checkCudaError(call)                                      \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void) {
    /* Allocate host memory */
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c = (float *)malloc(N * sizeof(float));
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host vectors\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    checkCudaError(cudaMalloc((void **)&d_a, N * sizeof(float)));
    checkCudaError(cudaMalloc((void **)&d_b, N * sizeof(float)));
    checkCudaError(cudaMalloc((void **)&d_c, N * sizeof(float)));

    /* Wrap HtoD for A */
    checkCudaError(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Wrap HtoD for B */
    checkCudaError(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Determine execution configuration */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    checkCudaError(cudaGetLastError()); // Check kernel launch

    /* Wrap DtoH for C */
    checkCudaError(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Verify a few results */
    for (int i = 0; i < 10; ++i) {
        printf("h_c[%d] = %f (expected %f)\n", i, h_c[i], h_a[i] + h_b[i]);
    }

    /* Clean up */
    free(h_a);
    free(h_b);
    free(h_c);
    checkCudaError(cudaFree(d_a));
    checkCudaError(cudaFree(d_b));
    checkCudaError(cudaFree(d_c));

    return EXIT_SUCCESS;
}
```