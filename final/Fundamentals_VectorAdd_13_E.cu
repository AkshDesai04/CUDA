```cuda
/*
Aim of the program:
Implement with `double` precision.

Thinking:
The user requested a CUDA program implemented with double precision.
Since no specific computation was specified, I chose a simple and common
CUDA example: element‑wise addition of two vectors of type double.
This demonstrates allocating host and device memory, copying data
between them, launching a kernel that uses double precision, and
measuring performance.

Key design points:
1. Use `double` everywhere (host arrays, device arrays, kernel arguments).
2. Kernel: `addKernel` adds elements from `A` and `B` into `C`.
3. Host code allocates memory, initializes arrays with random doubles,
   copies them to device, launches the kernel with enough blocks to
   cover all elements, copies the result back, and verifies correctness.
4. CUDA error checking is included for safety.
5. Simple timing with `cudaEvent_t` to show kernel execution time.

This code is self‑contained and can be compiled with:
    nvcc -arch=sm_50 -O2 -o double_vector_add double_vector_add.cu
and executed to see the result and timing.
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

/* CUDA kernel for element‑wise addition of double vectors */
__global__ void addKernel(const double *A, const double *B, double *C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

/* Helper macro for CUDA error checking */
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

/* Host function to generate random double values in range [0,1] */
void fillRandomDouble(double *array, size_t N)
{
    for (size_t i = 0; i < N; ++i) {
        array[i] = (double)rand() / RAND_MAX;
    }
}

/* Main host function */
int main(int argc, char **argv)
{
    /* Set vector size */
    const size_t N = (argc > 1) ? atoll(argv[1]) : 1 << 20; /* default 1M */
    const size_t bytes = N * sizeof(double);

    /* Allocate host memory */
    double *h_A = (double *)malloc(bytes);
    double *h_B = (double *)malloc(bytes);
    double *h_C = (double *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    srand((unsigned)time(NULL));
    fillRandomDouble(h_A, N);
    fillRandomDouble(h_B, N);

    /* Allocate device memory */
    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Define kernel launch parameters */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Timing events */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* Record start event */
    CHECK_CUDA(cudaEventRecord(start, 0));

    /* Launch kernel */
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Record stop event */
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    /* Calculate elapsed time */
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %f ms\n", ms);

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Simple verification: check first 10 elements */
    int errors = 0;
    for (int i = 0; i < 10 && i < N; ++i) {
        double expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-9) {
            fprintf(stderr, "Mismatch at index %zu: host=%f, device=%f\n",
                    i, expected, h_C[i]);
            errors++;
        }
    }
    if (errors == 0)
        printf("First 10 results verified.\n");
    else
        printf("%d mismatches found.\n", errors);

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
```