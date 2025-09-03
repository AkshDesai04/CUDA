```c
/*
Aim of the program: Implement for `float` data.

This CUDA program demonstrates a simple element‑wise vector addition using
`float` data type. The program creates two input arrays on the host, copies
them to the device, performs the addition on the GPU with a kernel launch,
and copies the result back to the host. The goal is to show how to manage
memory, launch a kernel, and handle basic error checking for floating‑point
operations in CUDA. The code is written in C and compiled with `nvcc`
as a single `.cu` file.

The thought process involved:
1. Choosing a straightforward example (vector addition) that showcases
   the essential CUDA workflow: memory allocation, data transfer,
   kernel execution, and result retrieval.
2. Using `float` to satisfy the requirement of implementing for float data.
3. Defining a kernel `vectorAdd` that processes each element independently.
4. Calculating grid and block dimensions to cover all elements.
5. Implementing robust error checking via a macro to catch CUDA API
   failures.
6. Initializing host data with simple values (e.g., A[i] = i, B[i] = 2*i)
   for easy verification of the output.
7. After copying the result back, printing the first few elements to
   confirm correctness.
8. Cleaning up all allocated memory on both host and device.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

/* Kernel for vector addition: C = A + B */
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20;           // Number of elements (1M)
    const int size = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = 2.0f * (float)i;
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Verify results */
    int errors = 0;
    for (int i = 0; i < 10; ++i) {  // check first 10 elements
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            printf("Mismatch at index %d: GPU %f, CPU %f\n", i, h_C[i], expected);
            errors++;
        }
    }
    if (errors == 0) {
        printf("All results are correct for the first 10 elements.\n");
    } else {
        printf("Found %d errors in the first 10 elements.\n", errors);
    }

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}
```