/*
Implement DAXPY, the double-precision version of the operation.

Thinking:
- The DAXPY operation is a classic linear algebra routine: y = a*x + y,
  where a is a scalar and x, y are vectors of doubles.
- For CUDA we need to allocate device memory for the vectors, copy data
  from host to device, launch a kernel that performs the operation
  in parallel, then copy the result back.
- Kernel design:
  * Use one thread per vector element.
  * Compute global index via blockIdx.x, blockDim.x, threadIdx.x.
  * Guard against out-of-bounds when N is not a multiple of block size.
- Grid/block configuration:
  * Choose a reasonable block size (e.g., 256 threads).
  * Compute grid size as (N + blockSize - 1) / blockSize.
- Error checking:
  * Wrap CUDA API calls in a macro to check errors.
  * Check kernel launch errors with cudaGetLastError() and
    cudaDeviceSynchronize().
- Host side:
  * Initialize sample data for x and y.
  * Use a small N (e.g., 1024) for easy debugging and printing.
  * After kernel execution, copy y back to host and print first few
    elements to verify correctness.
- Ensure double precision is used: use 'double' types everywhere
  and compile with nvcc which supports double precision on GPUs.
- The program is self‑contained and can be compiled with:
      nvcc -arch=sm_20 daxpy.cu -o daxpy
  (sm_20 is a safe baseline; newer GPUs will use a higher compute
  capability automatically.)
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

/* DAXPY kernel: y = a*x + y */
__global__ void daxpyKernel(double *y, const double *x, double a, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

int main(void)
{
    const int N = 1024;          /* Vector length */
    const double a = 2.5;        /* Scalar multiplier */

    /* Host memory allocation */
    double *h_x = (double *)malloc(N * sizeof(double));
    double *h_y = (double *)malloc(N * sizeof(double));
    if (!h_x || !h_y) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_x[i] = (double)i;
        h_y[i] = 2.0 * (double)i;   /* arbitrary init */
    }

    /* Device memory allocation */
    double *d_x = NULL;
    double *d_y = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&d_y, N * sizeof(double)));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    /* Launch DAXPY kernel */
    daxpyKernel<<<gridSize, blockSize>>>(d_y, d_x, a, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost));

    /* Print first 10 results for verification */
    printf("First 10 elements of y after DAXPY (y = a*x + y):\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("y[%d] = %f\n", i, h_y[i]);
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    free(h_x);
    free(h_y);

    return EXIT_SUCCESS;
}
