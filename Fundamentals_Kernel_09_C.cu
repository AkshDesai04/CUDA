/*
Aim: Explain the difference between `__global__` and `__device__` functions in a comment.

Thinking:
In CUDA, a `__global__` function is a kernel that can be launched from host (CPU) code. It runs on the device (GPU) and has a special launch syntax: `kernel<<<grid, block>>>(args)`. The compiler enforces that the function can only be called from host and that it has a void return type. It may invoke `__device__` functions or even other `__global__` kernels (through CUDA dynamic parallelism).

A `__device__` function, on the other hand, runs on the device and can only be called from other device or kernel code. It cannot be invoked from host. It can return a value, accept arguments, and can be inlined or called like a normal C function within device code.

To illustrate this difference we create a small program: a device function `add()` performs integer addition; a global kernel `kernel()` calls `add()` for each thread. The host allocates memory, launches the kernel, copies back results, and prints them. The program demonstrates that the kernel is launched from host, while the device function is only used within device code.*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors following a CUDA API call or kernel launch */
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

/* Device function: can only be called from device code */
__device__ int add(int a, int b) {
    return a + b;
}

/* Global kernel: launched from host */
__global__ void kernel(int *out, int val, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        /* Call device function */
        out[idx] = add(val, idx);
    }
}

int main(void) {
    const int N = 10;
    const int threadsPerBlock = 4;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int *h_out = (int*)malloc(N * sizeof(int));
    int *d_out = NULL;

    /* Allocate device memory */
    CHECK_CUDA(cudaMalloc((void**)&d_out, N * sizeof(int)));

    /* Launch kernel */
    kernel<<<blocks, threadsPerBlock>>>(d_out, 5, N);
    CHECK_CUDA(cudaGetLastError());

    /* Copy result back to host */
    CHECK_CCUDA(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Result of kernel (add 5 to each index):\n");
    for (int i = 0; i < N; ++i) {
        printf("h_out[%d] = %d\n", i, h_out[i]);
    }

    /* Clean up */
    free(h_out);
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}