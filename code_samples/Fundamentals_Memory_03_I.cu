```cuda
/*
Perform a three-way copy: HtoD to `d_a`, DtoD from `d_a` to `d_b`, DtoD from `d_b` to `d_c`. Verify `d_c`.

Thinking process:
1. Define array size and allocate host memory.
2. Initialize host array with known values.
3. Allocate three device arrays: d_a, d_b, d_c.
4. Copy host data to d_a (Host-to-Device).
5. Copy d_a to d_b (Device-to-Device).
6. Copy d_b to d_c (Device-to-Device).
7. Copy d_c back to a new host array h_c.
8. Verify that h_c matches the original host data.
9. Free all memory and handle CUDA errors with a macro for cleanliness.
10. Use synchronous copies for simplicity; no streams needed in this small example.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err)                                          \
    do {                                                         \
        cudaError_t e = (err);                                   \
        if (e != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA Error: %s (err=%d) at %s:%d\n", \
                    cudaGetErrorString(e), e, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                 \
        }                                                        \
    } while (0)

int main(void) {
    const size_t N = 1 << 20;           // 1M elements
    const size_t bytes = N * sizeof(int);

    // Allocate host memory
    int *h_a = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);
    if (!h_a || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = (int)i;
    }

    // Allocate device memory
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // HtoD copy to d_a
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // DtoD copy from d_a to d_b
    CUDA_CHECK(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));

    // DtoD copy from d_b to d_c
    CUDA_CHECK(cudaMemcpy(d_c, d_b, bytes, cudaMemcpyDeviceToDevice));

    // Copy d_c back to host for verification
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify result
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i]) {
            fprintf(stderr, "Mismatch at index %zu: expected %d, got %d\n",
                    i, h_a[i], h_c[i]);
            ok = false;
            break;
        }
    }

    if (ok) {
        printf("Verification passed: d_c matches original data.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Clean up
    free(h_a);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
```