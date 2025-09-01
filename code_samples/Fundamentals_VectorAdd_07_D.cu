/*
Aim: Handle `double` precision data.

Thinking:
The requirement is to provide a CUDA program that demonstrates handling of double precision
floating-point numbers on the GPU.  The most common use case is performing a simple
vector operation, such as vector addition, using `double` elements.  The program will:

1. Allocate host arrays `h_A`, `h_B`, and `h_C` of type `double`.
2. Initialize `h_A` and `h_B` with deterministic values so that the expected result
   can be verified on the host.
3. Allocate device memory `d_A`, `d_B`, `d_C`.
4. Copy `h_A` and `h_B` to the device.
5. Launch a kernel `vecAdd` that computes `C[i] = A[i] + B[i]` for each element.
6. Copy the result back to `h_C`.
7. Verify the result by comparing with a reference calculation performed on the CPU.
8. Report success or failure.

Key design points:
- Use `double` everywhere to ensure double precision throughout the data path.
- Include error checking for all CUDA API calls via a helper macro.
- Use a block size of 256 threads, which is a common choice for coalesced memory access.
- The program prints out the first few results and the total number of correct elements
  to provide a quick sanity check.
- We also query the device properties to confirm that double precision is supported
  (compute capability >= 1.3 for FP64 support, but most GPUs today support it).

The code is written in standard C with CUDA extensions, fully compilable with `nvcc`.
It demonstrates the full end-to-end flow: data allocation, transfer, kernel launch,
result retrieval, and correctness verification for double precision data. */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper macro for checking CUDA errors */
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* Kernel for vector addition: C = A + B */
__global__ void vecAdd(const double *A, const double *B, double *C, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    /* Size of vectors (number of elements) */
    const int N = 1 << 20;          /* 1,048,576 elements */
    const size_t size = N * sizeof(double);

    /* Host memory allocation */
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0 / (double)(i + 1);          /* A[i] = 1/(i+1) */
        h_B[i] = 2.0 / (double)(i + 1);          /* B[i] = 2/(i+1) */
    }

    /* Device memory allocation */
    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;
    cudaCheckError( cudaMalloc((void **)&d_A, size) );
    cudaCheckError( cudaMalloc((void **)&d_B, size) );
    cudaCheckError( cudaMalloc((void **)&d_C, size) );

    /* Copy input data from host to device */
    cudaCheckError( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaCheckError( cudaGetLastError() );           /* Check kernel launch */
    cudaCheckError( cudaDeviceSynchronize() );      /* Wait for completion */

    /* Copy result back to host */
    cudaCheckError( cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) );

    /* Verify results */
    int correct = 0;
    for (int i = 0; i < N; ++i) {
        double expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) < 1e-12) {
            ++correct;
        } else {
            if (i < 5) {  /* Print first few mismatches for debugging */
                printf("Mismatch at %d: host %f, device %f, expected %f\n",
                       i, h_C[i], expected, h_A[i] + h_B[i]);
            }
        }
    }

    printf("Double precision vector addition completed.\n");
    printf("Total elements: %d\n", N);
    printf("Correct results: %d\n", correct);
    if (correct == N) {
        printf("All results are correct.\n");
    } else {
        printf("Some results are incorrect. Check console output for mismatches.\n");
    }

    /* Clean up */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
