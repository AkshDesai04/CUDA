/*
Aim of the program:
Implement with double precision.

Thinking process:
- The user only gave a single line: "Implement with double precision."
- Since no specific functionality was requested, I decided to create a simple, illustrative CUDA program that demonstrates double‑precision arithmetic on the GPU.
- Vector addition is a classic example: it is straightforward, easy to understand, and requires only basic CUDA concepts (kernel launch, memory allocation, copying).
- The kernel will use the `double` type for all inputs, outputs, and arithmetic to satisfy the double‑precision requirement.
- The host code will:
  1. Allocate host arrays of `double`.
  2. Initialize them with sample values.
  3. Allocate device arrays and copy data to the GPU.
  4. Launch the kernel with a reasonable block/grid configuration.
  5. Copy the result back and verify the first few elements.
- All error checks will use `cudaGetLastError()` and `cudaMemcpy()` return codes.
- The program uses standard C syntax and CUDA runtime API, compiled with `nvcc` (e.g., `nvcc -arch=sm_35 -o vectorAdd vectorAdd.cu`).
- No external dependencies are required; the program is self‑contained and demonstrates double‑precision computation on the GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel that performs element‑wise addition of two double arrays */
__global__ void vectorAdd(const double *A, const double *B, double *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

/* Utility function to check CUDA errors */
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    const int N = 1 << 20;          // 1M elements
    const int bytes = N * sizeof(double);

    /* Allocate host memory */
    double *h_A = (double *)malloc(bytes);
    double *h_B = (double *)malloc(bytes);
    double *h_C = (double *)malloc(bytes);
    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host vectors\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = 1.0 * i;          // Example data
        h_B[i] = 2.0 * i;
    }

    /* Allocate device memory */
    double *d_A = NULL, *d_B = NULL, *d_C = NULL;
    checkCudaError(cudaMalloc((void **)&d_A, bytes), "cudaMalloc d_A");
    checkCudaError(cudaMalloc((void **)&d_B, bytes), "cudaMalloc d_B");
    checkCudaError(cudaMalloc((void **)&d_C, bytes), "cudaMalloc d_C");

    /* Copy data from host to device */
    checkCudaError(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D d_A");
    checkCudaError(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D d_B");

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    /* Copy result back to host */
    checkCudaError(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H d_C");

    /* Simple verification */
    for (int i = 0; i < 10; ++i)
    {
        printf("h_A[%d] = %f, h_B[%d] = %f, h_C[%d] = %f\n",
               i, h_A[i], i, h_B[i], i, h_C[i]);
    }

    /* Free device memory */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    /* Free host memory */
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
