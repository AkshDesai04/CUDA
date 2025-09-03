/*
Aim of the program: Handle `double` precision data.

Thinking:
The user simply asked for a program that handles double precision data in CUDA C. The safest interpretation is to provide a minimal, self‑contained example that demonstrates allocation, copying, and computation with double precision values on the device. 
A typical example is vector addition of two double‑precision arrays. 
The code will:
- Define the size of vectors (N)
- Allocate host arrays, fill them with sample values
- Allocate device memory for input and output vectors
- Copy inputs to the device
- Launch a kernel that adds corresponding elements using double precision
- Copy the result back to the host
- Verify the result
- Free all allocated memory

Key points:
- Use `double` type for all relevant variables.
- Use `cudaMalloc`, `cudaMemcpy`, and `cudaFree` for memory operations.
- The kernel will be very simple: `out[i] = a[i] + b[i];`
- We will use a grid of 256 threads per block for demonstration.
- After kernel execution, check for errors and print a success message.
- Include error checking for CUDA API calls.

This code is fully self‑contained and can be compiled with `nvcc` as `nvcc -arch=sm_20 -o vec_add_d vec_add_d.cu` (double precision requires compute capability 2.0+). 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Kernel to add two double vectors */
__global__ void vecAdd(const double *a, const double *b, double *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/* Utility macro for CUDA error checking */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

int main(void) {
    const int N = 1 << 20; /* 1M elements */
    size_t size = N * sizeof(double);

    /* Allocate host memory */
    double *h_a = (double*)malloc(size);
    double *h_b = (double*)malloc(size);
    double *h_c = (double*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (double)i;
        h_b[i] = (double)(N - i);
    }

    /* Allocate device memory */
    double *d_a = NULL;
    double *d_b = NULL;
    double *d_c = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    /* Copy host input to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* Verify results */
    bool success = true;
    for (int i = 0; i < N; ++i) {
        double expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-9) { /* tolerance for double precision */
            fprintf(stderr, "Mismatch at index %d: GPU %f != CPU %f\n", i, h_c[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition succeeded for %d elements with double precision.\n", N);
    } else {
        printf("Vector addition failed.\n");
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
