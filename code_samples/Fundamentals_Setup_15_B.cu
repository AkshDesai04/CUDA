```c
/*
Aim of the program:
Call `cudaDeviceReset()` at the very beginning of your `main` function to ensure a clean state.

Thought process:
The user explicitly wants a CUDA C program that demonstrates the use of cudaDeviceReset() at the start of main. 
I decided to create a minimal, self‑contained vector addition example because it illustrates a typical CUDA workflow 
and will compile and run on any machine with a CUDA toolkit installed. 

Key points to include:
1. A simple kernel that adds two input arrays.
2. Allocation of device memory, copying data from host to device, launching the kernel, copying results back, and freeing resources.
3. Comprehensive error checking with a helper macro to make debugging easier.
4. The call to cudaDeviceReset() is placed as the very first statement inside main to guarantee a clean device state before any other CUDA calls.
5. The code is written in C (using CUDA extensions) and is fully self‑contained, so no external files or libraries are required aside from the CUDA runtime.

This satisfies the requirement of showing how to reset the device at the beginning while also providing a working CUDA example.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for error checking */
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Simple element‑wise addition kernel */
__global__ void vecAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    /* Reset the device to ensure a clean state before any CUDA calls */
    CHECK_CUDA(cudaDeviceReset());

    /* Problem size */
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA(cudaMalloc((void**)&d_c, size));

    /* Copy input data to device */
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch errors

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* Verify results (simple check) */
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            if (errors < 10) {
                fprintf(stderr, "Mismatch at %d: expected %f, got %f\n",
                        i, expected, h_c[i]);
            }
            ++errors;
        }
    }
    if (errors == 0) {
        printf("Vector addition successful! No errors detected.\n");
    } else {
        printf("Vector addition completed with %d errors.\n", errors);
    }

    /* Clean up */
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    /* Reset device again before exiting (optional but clean) */
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
```