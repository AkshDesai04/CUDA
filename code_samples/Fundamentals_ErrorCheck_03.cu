/*
Aim: Kernel launches do not return an error code directly. Add error checking for a kernel launch by calling `cudaPeekAtLastError()` immediately after the launch and `cudaDeviceSynchronize()` sometime later. Wrap both of these calls with your `CHECK()` macro.

Thinking:  
The requirement is to demonstrate how to add error checking for a kernel launch in CUDA.  
In CUDA, a kernel launch does not return a CUDA error code directly; instead, any errors that occurred during the launch are stored and can be retrieved with `cudaPeekAtLastError()`.  After launching the kernel, we should immediately check this value.  Later, when the kernel has had time to execute, we should synchronize the device with `cudaDeviceSynchronize()`.  Any errors that happened during kernel execution will also be reported by this call.  
We wrap both calls with a `CHECK()` macro so that any error will be printed with file and line information and the program will exit.  
The example program below implements a simple vector addition kernel, launches it, performs the two error checks, and prints the result.  The code is self‑contained and can be compiled with `nvcc`.  It also includes basic memory allocation, copying, and cleanup.  The `CHECK()` macro is generic and can be reused for any CUDA runtime function that returns a `cudaError_t`.  
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check CUDA runtime API calls */
#define CHECK(call)                                              \
    do {                                                         \
        cudaError_t err = (call);                                \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

/* Simple kernel: element‑wise addition of two vectors */
__global__ void vecAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main(void)
{
    const int N = 1 << 20;   /* Number of elements (1M) */
    const int SIZE = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float *)malloc(SIZE);
    float *h_b = (float *)malloc(SIZE);
    float *h_c = (float *)malloc(SIZE);

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    CHECK(cudaMalloc((void **)&d_a, SIZE));
    CHECK(cudaMalloc((void **)&d_b, SIZE));
    CHECK(cudaMalloc((void **)&d_c, SIZE));

    /* Copy inputs to device */
    CHECK(cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, SIZE, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    /* Immediately check for launch errors */
    CHECK(cudaPeekAtLastError());

    /* Copy result back to host */
    CHECK(cudaMemcpy(h_c, d_c, SIZE, cudaMemcpyDeviceToHost));

    /* Synchronize and check for execution errors */
    CHECK(cudaDeviceSynchronize());

    /* Simple verification: print first 5 results */
    printf("First 5 results:\n");
    for (int i = 0; i < 5; ++i)
        printf("c[%d] = %f\n", i, h_c[i]);

    /* Clean up */
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    return 0;
}
