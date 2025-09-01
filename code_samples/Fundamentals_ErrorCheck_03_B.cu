/*
Create a `CHECK_KERNEL()` macro that specifically wraps these two calls for convenience.
The goal is to simplify CUDA kernel launch error checking. Typically a kernel launch is followed
by a call to `cudaGetLastError()` to capture any launch failures. Writing this boilerplate
repeatedly is error‑prone, so a macro that takes the kernel launch expression and automatically
performs the error check is useful. The macro should:
1. Execute the kernel launch expression.
2. Call `cudaGetLastError()` to retrieve the last error from the CUDA runtime.
3. If an error is detected, print a descriptive message (including file, line, and
   error string) and abort the program.
4. Be safe to use in any context (wrapped in a `do { ... } while(0)` block).
To achieve this, the macro accepts a single argument: the kernel launch statement,
which the user supplies without a trailing semicolon. Inside the macro we first
execute the launch, then perform the error check. An example usage is:
    CHECK_KERNEL(myKernel<<<blocks, threads>>>(args));
This expands to the kernel launch followed by the error checking logic. The macro
definition uses `__FILE__` and `__LINE__` to provide context in the error message.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Macro to launch a kernel and check for launch errors */
#define CHECK_KERNEL(kernel_launch)                                     \
    do {                                                                \
        kernel_launch;                                                 \
        cudaError_t err = cudaGetLastError();                          \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA kernel launch failed at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

/* Simple CUDA kernel that adds two arrays */
__global__ void addKernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1 << 20;
    size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }

    /* Allocate device memory */
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    /* Copy data from host to device */
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    /* Define kernel launch configuration */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel using the CHECK_KERNEL macro */
    CHECK_KERNEL(addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N));

    /* Copy result back to host */
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    /* Verify a few results */
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            ok = false;
            break;
        }
    }
    printf("Result verification: %s\n", ok ? "PASS" : "FAIL");

    /* Clean up */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
