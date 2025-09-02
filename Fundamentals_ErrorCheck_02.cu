/*
Aim of the program: Refactor your vector addition program from the previous section to use the `CHECK()` macro for every single CUDA API call (`cudaMalloc`, `cudaMemcpy`, `cudaFree`, etc.).

Thinking Process:
1. Define a macro `CHECK()` that wraps a CUDA API call, checks its return value, and prints a detailed error message if the call fails. This macro will also exit the program to avoid cascading failures.
2. Write a simple vector addition kernel that adds two input arrays element‑wise and stores the result in a third array.
3. In the host code, allocate host memory for the input and output vectors, initialize the inputs, and allocate device memory for each vector.
4. Replace every CUDA API call (`cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`, `cudaFree`) with the `CHECK()` macro so that failures are caught immediately.
5. Launch the kernel, synchronize, copy the result back to host, free all allocated memory, and print a confirmation message.
6. The program is written in standard C with CUDA extensions, compiled as a `.cu` file.

This ensures that any error in memory allocation, data transfer, kernel launch, or deallocation is reported with context, making debugging easier and the code robust.

The final code below follows these steps.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check CUDA API calls and report errors */
#define CHECK(call)                                                     \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s (%s)\n",           \
                    __FILE__, __LINE__,                                \
                    cudaGetErrorString(err),                           \
                    cudaGetErrorName(err));                            \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* Kernel for vector addition: C = A + B */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    /* Problem size */
    const int N = 1 << 20; /* 1 million elements */
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CHECK(cudaMalloc((void **)&d_A, size));
    CHECK(cudaMalloc((void **)&d_B, size));
    CHECK(cudaMalloc((void **)&d_C, size));

    /* Copy input data from host to device */
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Launch kernel: 256 threads per block */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    CHECK(cudaGetLastError());
    /* Wait for kernel to finish */
    CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Verify results (optional) */
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            if (errors < 10) {
                fprintf(stderr, "Mismatch at index %d: GPU %f != CPU %f\n",
                        i, h_C[i], expected);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("Vector addition successful, no errors detected.\n");
    } else {
        printf("Vector addition completed with %d errors.\n", errors);
    }

    /* Free device memory */
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    /* Free host memory */
    free(h_A);
    free(h_B);
    free(h_C);

    /* Reset device */
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
