```cpp
/* 
Aim of the program: 
Create a CHECK_KERNEL() macro that specifically wraps these two calls for convenience. 

Thinking process: 
- The usual pattern when launching a CUDA kernel is to invoke the kernel, then check for launch errors using cudaGetLastError(), and optionally synchronize the device to catch runtime errors with cudaDeviceSynchronize().
- The macro should simplify this pattern so that a user can simply write CHECK_KERNEL(grid, block, kernel, args...) instead of writing both calls manually.
- To make the macro flexible, we accept grid and block dimensions as parameters, followed by the kernel name and its arguments. 
- Inside the macro we perform the kernel launch, then call cudaGetLastError(). If an error occurs, we print a message and exit. 
- We also call cudaDeviceSynchronize() to catch any errors that occur during kernel execution. If that fails, we again print a message and exit.
- The macro uses a do { ... } while(0) construct to ensure it behaves like a single statement. 
- The macro expands to a block of code that includes error handling for both the launch and the execution phases, thereby wrapping the two calls in a convenient single invocation. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* 
 * Example kernel that adds two arrays.
 */
__global__ void add_arrays(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

/*
 * Macro that wraps kernel launch and error checking.
 * Parameters:
 *   grid   : dim3 grid dimensions
 *   block  : dim3 block dimensions
 *   kernel : kernel function name
 *   ...    : kernel arguments
 */
#define CHECK_KERNEL(grid, block, kernel, ...)                      \
    do {                                                           \
        kernel<<<grid, block>>>(__VA_ARGS__);                      \
        cudaError_t err = cudaGetLastError();                      \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "Kernel launch error: %s\n",           \
                    cudaGetErrorString(err));                      \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
        err = cudaDeviceSynchronize();                             \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "Kernel execution error: %s\n",         \
                    cudaGetErrorString(err));                      \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while(0)

/*
 * Host function that demonstrates the use of CHECK_KERNEL.
 */
int main(void)
{
    const int N = 1 << 20;          // 1M elements
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    /* Copy data to device */
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    /* Define grid and block dimensions */
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    /* Launch kernel with error checking */
    CHECK_KERNEL(grid, block, add_arrays, d_a, d_b, d_c, N);

    /* Copy result back to host */
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    /* Verify result */
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            ok = false;
            printf("Mismatch at %d: got %f, expected %f\n", i, h_c[i], expected);
            break;
        }
    }
    if (ok) {
        printf("Result is correct.\n");
    }

    /* Clean up */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```