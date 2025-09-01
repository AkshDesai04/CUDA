```cuda
/* 
Aim of the program: Verify the `double` precision results on the CPU. Note the potential for tiny floating point discrepancies.

Thinking process:
- The goal is to compare a double-precision computation performed on the host (CPU) with an equivalent computation performed on the device (GPU).
- We choose a simple but non-trivial operation: summing the squares of an array of double values. This involves many floating-point operations and can reveal small discrepancies due to different accumulation orders or rounding differences.
- The CPU will compute the sum in a straightforward loop.
- The GPU will use a parallel reduction kernel:
  * Each thread block will compute a partial sum of its assigned portion of the array using shared memory.
  * Each block writes its partial sum to a global array.
  * After kernel execution, the host gathers these partial sums and performs a final reduction on the CPU to produce the final GPU result.
- We use CUDA atomic addition for double precision only if the compute capability supports it; otherwise we avoid it and perform the final accumulation on the host. To keep the code portable, we will do the final accumulation on the host.
- After obtaining both results, we compare them. Because floating-point addition is not associative, small differences are expected. We compute an absolute difference and a relative error. If the relative error is below a tolerance (e.g., 1e-12), we consider the results effectively equal.
- We print both results, the absolute and relative differences, and whether the results are considered equal within the chosen tolerance.
- The program is self-contained, uses standard CUDA runtime API calls, and compiles with `nvcc`. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define BLOCK_SIZE 256

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel to compute partial sums of squares
__global__ void sum_of_squares(const double *x, double *partial, int n)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    double sum = 0.0;

    // Stride across the array to cover all elements
    while (i < n) {
        sum += x[i] * x[i];
        i += gridDim.x * blockDim.x;
    }

    // Load partial sum into shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduction within the block
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the block's partial sum to global memory
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

int main()
{
    // Problem size
    const size_t N = 1 << 24; // 16 million elements
    const size_t ARRAY_BYTES = N * sizeof(double);

    // Allocate host memory
    double *h_x = (double *)malloc(ARRAY_BYTES);
    if (!h_x) {
        fprintf(stderr, "Failed to allocate host array.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with pseudo-random doubles
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < N; ++i) {
        h_x[i] = (double)rand() / RAND_MAX; // values in [0,1]
    }

    // CPU computation
    double cpu_sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        cpu_sum += h_x[i] * h_x[i];
    }

    // Allocate device memory
    double *d_x = NULL;
    double *d_partial = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_x, ARRAY_BYTES));

    // Determine kernel launch configuration
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    size_t partialBytes = gridDim.x * sizeof(double);

    CUDA_CHECK(cudaMalloc((void **)&d_partial, partialBytes));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, ARRAY_BYTES, cudaMemcpyHostToDevice));

    // Launch kernel
    sum_of_squares<<<gridDim, blockDim, BLOCK_SIZE * sizeof(double)>>>(d_x, d_partial, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums back to host
    double *h_partial = (double *)malloc(partialBytes);
    if (!h_partial) {
        fprintf(stderr, "Failed to allocate host partial array.\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, partialBytes, cudaMemcpyDeviceToHost));

    // Final reduction on host to obtain GPU sum
    double gpu_sum = 0.0;
    for (size_t i = 0; i < gridDim.x; ++i) {
        gpu_sum += h_partial[i];
    }

    // Compare results
    double abs_diff = fabs(cpu_sum - gpu_sum);
    double rel_error = abs_diff / fabs(cpu_sum);

    const double tolerance = 1e-12; // relative tolerance

    printf("CPU sum:  %.15f\n", cpu_sum);
    printf("GPU sum:  %.15f\n", gpu_sum);
    printf("Absolute difference: %.15e\n", abs_diff);
    printf("Relative error:      %.15e\n", rel_error);
    if (rel_error < tolerance) {
        printf("Result: Within tolerance (%.2e < %.2e). OK.\n", rel_error, tolerance);
    } else {
        printf("Result: Exceeds tolerance. Potential discrepancy.\n");
    }

    // Clean up
    free(h_x);
    free(h_partial);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_partial));

    return 0;
}
```