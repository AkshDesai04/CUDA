/*
Implement element-wise vector multiplication: C[i] = A[i] * B[i].
This is the aim of the program.
[Thinking]
To perform element-wise multiplication of two vectors on a GPU using CUDA, the program follows these steps:

1. **Kernel Design**  
   - A simple CUDA kernel `vectorMultiply` will be written.  
   - Each thread computes one element `C[i] = A[i] * B[i]`.  
   - Thread index calculation uses `blockIdx`, `blockDim`, and `threadIdx`.  
   - Bounds checking ensures that threads with an index greater than the vector length do nothing.

2. **Host Setup**  
   - Vector size `N` is defined (e.g., 1<<20).  
   - Host arrays `h_A`, `h_B`, and `h_C` are allocated and initialized.  
   - Device arrays `d_A`, `d_B`, and `d_C` are allocated on the GPU.  
   - Host data is copied to device memory using `cudaMemcpy`.

3. **Execution Configuration**  
   - A typical block size of 256 threads is used; grid size is computed as `(N + blockSize - 1) / blockSize`.

4. **Kernel Launch**  
   - The kernel is launched with the computed grid and block dimensions.

5. **Result Retrieval and Validation**  
   - After kernel execution, `d_C` is copied back to `h_C`.  
   - For correctness, the program checks a few elements (or all) and prints any mismatches.

6. **Cleanup**  
   - All device memory is freed, and the program exits gracefully.

7. **Error Checking**  
   - Utility function `checkCuda` wraps CUDA API calls to handle errors cleanly.

The code is fully self‑contained, compiles with `nvcc`, and demonstrates the core idea of parallel vector multiplication on the GPU.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Utility macro for CUDA error checking */
#define checkCuda(call)                                                     \
    {                                                                       \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    }

/* CUDA kernel for element-wise vector multiplication */
__global__ void vectorMultiply(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}

int main(void) {
    /* Vector size */
    const int N = 1 << 20; /* 1,048,576 elements */
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors with sample data */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc((void **)&d_A, size));
    checkCuda(cudaMalloc((void **)&d_B, size));
    checkCuda(cudaMalloc((void **)&d_C, size));

    /* Copy data from host to device */
    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Determine execution configuration */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    checkCuda(cudaGetLastError());

    /* Copy result back to host */
    checkCuda(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    int errors = 0;
    for (int i = 0; i < N; i += N / 10) { /* sample 10 points */
        float expected = h_A[i] * h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: GPU=%f, CPU=%f\n", i, h_C[i], expected);
            errors++;
            if (errors > 10) break; /* limit output */
        }
    }

    if (errors == 0) {
        printf("All results are correct.\n");
    } else {
        printf("Total mismatches: %d\n", errors);
    }

    /* Clean up */
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
