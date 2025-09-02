```cpp
/*
Implement vector addition for `double` precision floating-point numbers instead of `float`s.

Thoughts:
- Change kernel signature and types to double.
- Allocate device memory for double using cudaMalloc with size * sizeof(double).
- Initialize host arrays with double values.
- Copy to device and back using cudaMemcpy.
- Launch kernel with proper grid dimensions.
- Verify by checking difference between C[i] and A[i]+B[i].
- Use cudaEvent for timing.
- Ensure error checking with cudaGetLastError and cudaDeviceSynchronize.
- Use a reasonably large vector size, e.g., 1<<20 (1M).
- Use C-style I/O and memory allocation to keep the code in C.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Vector addition kernel for double precision
__global__ void vectorAdd(const double *A, const double *B, double *C, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

// Error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",        \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),            \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

int main(void)
{
    const int N = 1 << 20; // 1 million elements
    const size_t bytes = N * sizeof(double);

    // Allocate host memory
    double *h_A = (double *)malloc(bytes);
    double *h_B = (double *)malloc(bytes);
    double *h_C = (double *)malloc(bytes);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0 * i;
        h_B[i] = 2.0 * i;
    }

    // Allocate device memory
    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        double expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-9) {
            fprintf(stderr, "Mismatch at index %d: GPU %f != CPU %f\n", i, h_C[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition successful. First 10 results:\n");
        for (int i = 0; i < 10; ++i)
            printf("C[%d] = %f\n", i, h_C[i]);
    } else {
        printf("Vector addition failed.\n");
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```