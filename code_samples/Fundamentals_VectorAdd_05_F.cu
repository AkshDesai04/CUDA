/*
Aim of the program: Handle large vectors with boundary checks.

Thinking Process:
1. The requirement is to write a CUDA program in C that handles large vectors safely by including boundary checks within the kernel to prevent out-of-bounds memory accesses.
2. A simple but representative operation is vector addition (C = A + B). This demonstrates memory allocation, transfer, kernel launch, and result verification.
3. To accommodate large vectors, the code will use dynamic allocation and a sizable vector size (e.g., 1<<20 elements, which is 1,048,576). The vector size can easily be increased to test scalability.
4. The kernel will calculate a global thread index and check if it is less than the vector length. If so, it performs the addition; otherwise it does nothing. This is the core boundary check.
5. The host code will:
   - Allocate and initialize two host vectors.
   - Allocate device memory.
   - Copy data from host to device.
   - Determine grid and block dimensions based on the vector size.
   - Launch the kernel.
   - Copy the result back to host.
   - Verify the result by checking a few sample elements.
   - Free allocated memory.
6. Error checking is performed after each CUDA call to catch issues early.
7. The code is fully self‑contained and can be compiled with `nvcc` (e.g., `nvcc -o vector_add vector_add.cu`).
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for CUDA error checking */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Kernel performing element-wise addition of two vectors with boundary checks */
__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    /* Vector size: 1<<20 (1,048,576) elements */
    const size_t N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors with sample data */
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    /* Copy input data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Simple verification: print first, middle, last elements */
    printf("Verification samples:\n");
    printf("h_C[0] = %.1f (expected %.1f)\n", h_C[0], h_A[0] + h_B[0]);
    printf("h_C[N/2] = %.1f (expected %.1f)\n", h_C[N/2], h_A[N/2] + h_B[N/2]);
    printf("h_C[N-1] = %.1f (expected %.1f)\n", h_C[N-1], h_A[N-1] + h_B[N-1]);

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    /* Free host memory */
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
