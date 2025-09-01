/*
Aim: Add a conditional compilation flag (e.g., `_DEBUG`) so the `CHECK` macro only performs the check in debug builds, and is empty in release builds for performance.

Thinking:
- The goal is to make a `CHECK` macro that evaluates a condition only when a debug flag (`_DEBUG`) is defined.
- In release mode, where `_DEBUG` is not defined, the macro should expand to nothing to avoid runtime overhead.
- This macro will be used to guard CUDA API error checks and any other runtime checks.
- The program demonstrates a simple vector addition kernel. All CUDA API calls are wrapped with `CHECK` to illustrate conditional debugging.
- The program is self‑contained, compilable with `nvcc`, and prints a success message when all checks pass in debug mode.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifdef _DEBUG
#define CHECK(condition)                                               \
    do {                                                               \
        if (!(condition)) {                                           \
            fprintf(stderr, "CHECK failed: %s, file %s, line %d\n",   \
                    #condition, __FILE__, __LINE__);                  \
            exit(EXIT_FAILURE);                                       \
        }                                                              \
    } while (0)
#else
#define CHECK(condition) do { } while (0)
#endif

__global__ void add_kernel(const float *a, const float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1 << 20;               // 1M elements
    size_t size = N * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, size) == cudaSuccess);
    CHECK(cudaMalloc(&d_b, size) == cudaSuccess);
    CHECK(cudaMalloc(&d_c, size) == cudaSuccess);

    CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice) == cudaSuccess);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK(cudaGetLastError() == cudaSuccess);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);

    CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost) == cudaSuccess);

    // Verify results
    for (int i = 0; i < N; ++i) {
        CHECK(h_c[i] == h_a[i] + h_b[i]);
    }

    printf("All checks passed.\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
