/*
Aim of the program:
Refactor the host code into functions: `init_vectors()`, `run_kernel()`, `verify_results()`

Thinking:
- The goal is to split the existing CUDA host code into three distinct functions to improve readability and modularity.
- `init_vectors()` will handle allocation of host memory and initialization of the input vectors.
- `run_kernel()` will allocate device memory, copy data from host to device, launch the kernel, and copy results back to host.
- `verify_results()` will compare the computed results with the expected values and report any discrepancies.
- The main function will simply orchestrate the calls to these functions.
- We'll implement a simple vector addition kernel as an example.
- Memory cleanup will be handled at the end of main.
- Error checking will be included for CUDA API calls.
- The code is self‑contained and ready to compile with `nvcc`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void init_vectors(float **h_A, float **h_B, float **h_C, size_t N)
{
    *h_A = (float *)malloc(N * sizeof(float));
    *h_B = (float *)malloc(N * sizeof(float));
    *h_C = (float *)malloc(N * sizeof(float));
    if (!*h_A || !*h_B || !*h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < N; ++i) {
        (*h_A)[i] = (float)i;
        (*h_B)[i] = (float)(N - i);
    }
}

void run_kernel(const float *h_A, const float *h_B, float *h_C, size_t N)
{
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;

    CHECK_CUDA(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_B, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_C, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

void verify_results(const float *h_A, const float *h_B, const float *h_C, size_t N)
{
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            fprintf(stderr, "Verification failed at index %zu: "
                            "host %f, device %f, expected %f\n",
                    i, h_C[i], expected, h_C[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Verification passed: all results are correct.\n");
    } else {
        printf("Verification failed.\n");
    }
}

int main(void)
{
    const size_t N = 1 << 20; // 1M elements
    float *h_A, *h_B, *h_C;

    init_vectors(&h_A, &h_B, &h_C, N);
    run_kernel(h_A, h_B, h_C, N);
    verify_results(h_A, h_B, h_C, N);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
