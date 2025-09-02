```c
/*
Add error checking to the `cudaMalloc` and `cudaMemcpy` calls.

Thinking process:
- The goal is to demonstrate how to add robust error handling for CUDA memory allocation
  and memory copy operations. I want to keep the code simple yet functional, so I will
  write a small vector addition kernel that copies data from host to device, performs
  a computation, and copies results back to the host.
- I will define a helper macro `CHECK_CUDA` that wraps any CUDA runtime API call,
  checks its return value, prints a detailed error message (including file, line,
  and error string), and aborts the program if an error occurs.
- The macro will be used for both `cudaMalloc` and `cudaMemcpy`. Additionally,
  I will add a check after the kernel launch with `cudaGetLastError` and
  `cudaDeviceSynchronize` to catch launch failures.
- This example will allocate arrays on the device, copy data from host arrays
  to device, launch a simple kernel that adds two vectors, copy the result
  back to the host, and then free the device memory. Throughout, each CUDA call
  will be wrapped with the error checking macro.
- The program is written in standard C (with CUDA extensions) and is fully
  self-contained. It can be compiled with `nvcc` and run on any system with
  a CUDA-capable GPU.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Simple vector addition kernel */
__global__ void vecAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main(void)
{
    const int N = 1 << 20;  /* 1M elements */
    size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize host data */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory with error checking */
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_a, size));
    CHECK_CUDA(cudaMalloc((void **)&d_b, size));
    CHECK_CUDA(cudaMalloc((void **)&d_c, size));

    /* Copy data from host to device with error checking */
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    /* Kernel launch configuration */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel and check for errors */
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    /* Check for launch errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    /* Synchronize and check for errors during execution */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Copy result back to host with error checking */
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    for (int i = 0; i < 5; ++i) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }

    /* Clean up */
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    printf("Vector addition completed successfully.\n");
    return EXIT_SUCCESS;
}
```