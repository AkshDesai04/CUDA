/*
Aim: Allocate `d_a` and `d_b`. Use `cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice)` to initialize `d_a`. What is the error? Change `h_a` to `d_a`. What is the error? Explain why the kind parameter is critical.

Thinking:
- The program demonstrates common mistakes with cudaMemcpy.
- First, we allocate two device arrays d_a and d_b, and one host array h_a.
- We correctly copy from h_a to d_b using cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice). This compiles and runs fine, but it actually initializes d_b, not d_a. So d_a remains uninitialized. If we later try to read from d_a, we will get garbage or undefined behavior. This is a logical error, not a runtime error from cudaMemcpy itself.
- Next, we attempt to copy from d_a to d_b using cudaMemcpy(d_b, d_a, size, cudaMemcpyHostToDevice). Here the source pointer d_a is a device pointer, but the copy kind is cudaMemcpyHostToDevice which expects a host source. The CUDA runtime will detect the mismatch and return an error (cudaErrorInvalidMemcpyDirection). This shows that the `kind` parameter is critical: it tells the runtime which memory spaces the pointers belong to. Using the wrong kind causes the call to fail.
- The code will print the error codes after each cudaMemcpy to illustrate the differences.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10
#define CHECK_CUDA(call)                                                     \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main(void)
{
    size_t size = N * sizeof(int);

    /* Host array */
    int h_a[N];
    for (int i = 0; i < N; ++i) h_a[i] = i;

    /* Device pointers */
    int *d_a = NULL;
    int *d_b = NULL;

    /* Allocate device memory */
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));

    /* ---- First copy: Intended to initialize d_a but copies to d_b ---- */
    printf("Copying from host h_a to device d_b (intended to init d_a)\n");
    CHECK_CUDA(cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice));
    printf("Result of first cudaMemcpy: %s\n",
           cudaGetErrorString(cudaGetLastError()));

    /* At this point, d_a is still uninitialized. Reading it would be undefined. */

    /* ---- Second copy: Using d_a as source with HostToDevice kind ---- */
    printf("\nAttempting to copy from device d_a to device d_b with cudaMemcpyHostToDevice kind\n");
    cudaError_t err = cudaMemcpy(d_b, d_a, size, cudaMemcpyHostToDevice);
    printf("Result of second cudaMemcpy: %s\n", cudaGetErrorString(err));

    /* Clean up */
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    return 0;
}
