```cpp
/*
Create a `CHECK_KERNEL()` macro that specifically wraps these two calls for convenience.

Thinking:
The prompt asks for a macro that wraps "these two calls". In the context of CUDA programming, a common pattern is to launch a kernel and then immediately check for launch errors with cudaGetLastError() and execution errors with cudaDeviceSynchronize(). 
Wrapping these two calls into a single macro reduces boilerplate and makes error handling concise. 

The macro will:
1. Perform the kernel launch with the provided launch configuration and arguments.
2. Capture any launch errors via cudaGetLastError().
3. If an error occurs, print a descriptive message and abort.
4. Synchronize the device with cudaDeviceSynchronize() to catch any runtime errors.
5. Again, check the return status and handle failures.

The macro needs to accept the kernel function, launch configuration (grid/block), and arguments. 
For flexibility, we will pass the kernel call and its configuration as a single argument (the expression ending with `<<<...>>>`). 
The macro will then perform the error checks. 
Example usage:
    CHECK_KERNEL(myKernel<<<blocks, threads>>>(arg1, arg2));
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Macro to launch a kernel and check for both launch and execution errors */
#define CHECK_KERNEL(kernel_call)                                        \
    do {                                                                 \
        kernel_call;                                                    \
        cudaError_t err = cudaGetLastError();                           \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "Kernel launch failed: %s\n",                \
                    cudaGetErrorString(err));                           \
            exit(EXIT_FAILURE);                                         \
        }                                                                \
        err = cudaDeviceSynchronize();                                  \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "Kernel execution failed: %s\n",             \
                    cudaGetErrorString(err));                           \
            exit(EXIT_FAILURE);                                         \
        }                                                                \
    } while (0)

/* Simple vector addition kernel */
__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    /* Copy inputs to device */
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    /* Determine launch configuration */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel with CHECK_KERNEL macro */
    CHECK_KERNEL( vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N) );

    /* Copy result back to host */
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    /* Verify result for a few elements */
    for (int i = 0; i < 5; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    /* Clean up */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```