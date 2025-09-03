```c
/*
Refactor the host code into functions: init_vectors(), run_kernel(), verify_results()

Thinking:
- The goal is to split the host-side logic of a CUDA program into three distinct functions.
- `init_vectors()` will allocate and populate the host input arrays.
- `run_kernel()` will handle all device memory allocation, data transfer, kernel launch, and result copy back.
- `verify_results()` will compare the device output with the expected host-side result.
- A simple vector addition kernel is used to keep the example clear and focused on the refactoring task.
- Error checking is minimal for brevity; the example assumes a CUDA-capable system and that `N` is a reasonably small size.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 1024            // Size of the vectors
#define THREADS_PER_BLOCK 256

/* CUDA kernel: element‑wise addition of two vectors */
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

/* ------------------------------------------------------------------ */
/* init_vectors: Allocate host memory and fill the input arrays       */
/* ------------------------------------------------------------------ */
void init_vectors(float **h_A, float **h_B, float **h_C) {
    *h_A = (float*)malloc(N * sizeof(float));
    *h_B = (float*)malloc(N * sizeof(float));
    *h_C = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        (*h_A)[i] = rand() / (float)RAND_MAX;
        (*h_B)[i] = rand() / (float)RAND_MAX;
    }
}

/* ------------------------------------------------------------------ */
/* run_kernel: Allocate device memory, copy data, launch kernel, and  */
/*             copy results back to host                             */
/* ------------------------------------------------------------------ */
void run_kernel(float *h_A, float *h_B, float *h_C) {
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;

    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/* ------------------------------------------------------------------ */
/* verify_results: Compare device output with expected host result    */
/* ------------------------------------------------------------------ */
void verify_results(float *h_A, float *h_B, float *h_C) {
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            printf("Mismatch at index %d: expected %f, got %f\n",
                   i, expected, h_C[i]);
            return;
        }
    }
    printf("All results verified successfully.\n");
}

/* ------------------------------------------------------------------ */
/* main: Orchestrate the workflow using the refactored functions      */
/* ------------------------------------------------------------------ */
int main() {
    srand((unsigned)time(NULL));

    float *h_A, *h_B, *h_C;

    init_vectors(&h_A, &h_B, &h_C);
    run_kernel(h_A, h_B, h_C);
    verify_results(h_A, h_B, h_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```