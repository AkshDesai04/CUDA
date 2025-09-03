/*
Aim of the program: Implement with `double` precision.

Thought process:
1. The user explicitly requests a CUDA program in C that uses double precision.
2. No specific algorithm is mentioned, so I chose a classic example that demonstrates double precision operations: vector addition.
3. The program will:
   - Allocate two input vectors A and B and an output vector C on the host.
   - Allocate corresponding device memory.
   - Copy A and B to the device.
   - Launch a kernel that performs element-wise addition using `double` types.
   - Copy the result back to the host.
   - Verify a few elements and print success or an error message.
4. I included basic error checking with a helper macro to keep the code concise.
5. The program is fully self‑contained and can be compiled with `nvcc` to produce an executable.

Compile with:
    nvcc -arch=sm_52 vector_add_double.cu -o vector_add_double

Run:
    ./vector_add_double
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper macro for error checking */
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Kernel that performs element-wise addition of two double vectors */
__global__ void vectorAdd(const double *A, const double *B, double *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    /* Problem size */
    const int N = 1 << 20;          // 1M elements
    const size_t bytes = N * sizeof(double);

    /* Allocate host memory */
    double *h_A = (double *)malloc(bytes);
    double *h_B = (double *)malloc(bytes);
    double *h_C = (double *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (double)i;
        h_B[i] = 2.0 * (double)i;
    }

    /* Allocate device memory */
    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    /* Copy inputs to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify a few elements */
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        double expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-12) {
            fprintf(stderr, "Mismatch at index %d: host %f, device %f\n",
                    i, expected, h_C[i]);
            ++errors;
        }
    }
    if (errors == 0) {
        printf("Vector addition successful. First 10 results:\n");
        for (int i = 0; i < 10; ++i) {
            printf("C[%d] = %f\n", i, h_C[i]);
        }
    } else {
        printf("%d errors found in vector addition.\n", errors);
    }

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
