/*
Aim of the program: Verify the results on the CPU.

Thinking process:

The task is to create a CUDA program that performs a simple parallel operation on the GPU and then verifies the results on the CPU. Vector addition is a common, straightforward example that demonstrates GPU computing and correctness checks. The program will:

1. Allocate two input vectors (A and B) and an output vector (C) on the host.
2. Initialize the input vectors with deterministic values (e.g., A[i] = i, B[i] = 2*i) so the expected result is known.
3. Allocate corresponding memory on the device (d_A, d_B, d_C).
4. Copy the input data from host to device.
5. Launch a CUDA kernel that adds A[i] + B[i] and stores the sum in C[i].
6. Copy the result vector back to the host.
7. Verify the result on the CPU by comparing each element to the expected value.
8. Report success or failure.

Error checking is added for CUDA API calls and kernel launches to ensure any failures are caught early. The code is written in C with CUDA extensions and can be compiled with nvcc. It includes necessary headers, macro definitions, kernel, main function, and cleanup.

The program is self-contained and can be run on any system with CUDA support. It prints a success message if all elements match the expected results; otherwise, it reports a mismatch.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 20)  // Number of elements (1M)
#define THREADS_PER_BLOCK 256

/* Error checking macro */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Kernel to perform element-wise addition: C = A + B */
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_C = NULL;
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    size_t size = N * sizeof(float);

    /* Allocate host memory */
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = 2.0f * (float)i;
    }

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    /* Copy input data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Determine grid dimensions */
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    /* Launch kernel */
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Verify results on CPU */
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            if (errors < 10) { // Print first few errors
                fprintf(stderr, "Mismatch at index %d: GPU=%f, CPU=%f\n",
                        i, h_C[i], expected);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("All %d results are correct. Verification succeeded.\n", N);
    } else {
        printf("Verification failed: %d mismatches found.\n", errors);
    }

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    /* Free host memory */
    free(h_A);
    free(h_B);
    free(h_C);

    /* Reset device (optional) */
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
