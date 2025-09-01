```c
/* 
Aim of the program: Verify the results of multiplication on the CPU. 
Thought process:

The task is to confirm that a multiplication operation performed on the GPU
produces the same results as when the same operation is carried out on the
CPU.  A straightforward way to do this is to multiply two vectors element‑wise
on the GPU, copy the result back to host memory, compute the same multiplication
on the CPU, and then compare the two results.  If all corresponding elements
match (within a small numerical tolerance), the verification passes.

Implementation steps:
1. Define a CUDA kernel that multiplies two input vectors element‑wise.
2. In main:
   - Parse optional command‑line argument for vector size; default to 1024.
   - Allocate host memory for input vectors `h_a`, `h_b` and result vectors
     `h_c_gpu` (GPU output) and `h_c_cpu` (CPU computed).
   - Initialize the input vectors with deterministic values.
   - Allocate device memory for `d_a`, `d_b`, `d_c`.
   - Copy `h_a` and `h_b` to device.
   - Launch the kernel with a reasonable grid/block configuration.
   - Copy the result back into `h_c_gpu`.
   - Compute the multiplication on the CPU into `h_c_cpu`.
   - Compare `h_c_gpu` and `h_c_cpu` element‑wise with a tolerance of
     `1e-5f`.  Report success or the first mismatch found.
   - Free all allocated memory.
3. Add basic error checking for CUDA API calls.

The program is self‑contained and can be compiled with `nvcc`:
    nvcc -o vector_mul_verify vector_mul_verify.cu
It can optionally take the vector size as an argument:
    ./vector_mul_verify 1048576
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* CUDA kernel for element‑wise multiplication of two vectors */
__global__ void vectorMul(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] * b[idx];
}

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main(int argc, char *argv[])
{
    /* Vector size */
    int N = 1024; /* default size */
    if (argc >= 2) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid vector size.\n");
            return EXIT_FAILURE;
        }
    }

    /* Allocate host memory */
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c_gpu = (float *)malloc(N * sizeof(float));
    float *h_c_cpu = (float *)malloc(N * sizeof(float));

    if (!h_a || !h_b || !h_c_gpu || !h_c_cpu) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input vectors with deterministic values */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)(i + 1);          // 1.0f, 2.0f, ...
        h_b[i] = 2.0f * (float)(i + 1);   // 2.0f, 4.0f, ...
    }

    /* Allocate device memory */
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_c, N * sizeof(float)));

    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());  // Check kernel launch error

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Compute multiplication on CPU */
    for (int i = 0; i < N; ++i) {
        h_c_cpu[i] = h_a[i] * h_b[i];
    }

    /* Verify results */
    const float tol = 1e-5f;
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        float diff = fabsf(h_c_gpu[i] - h_c_cpu[i]);
        if (diff > tol) {
            if (mismatches < 10) {  // Report first few mismatches
                printf("Mismatch at index %d: GPU=%f, CPU=%f, diff=%f\n",
                       i, h_c_gpu[i], h_c_cpu[i], diff);
            }
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("Verification PASSED: All results match within tolerance %e.\n", tol);
    } else {
        printf("Verification FAILED: %d mismatches found.\n", mismatches);
    }

    /* Clean up */
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
```