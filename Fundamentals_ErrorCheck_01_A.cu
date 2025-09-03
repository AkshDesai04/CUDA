```cpp
/*
 * Aim: Make the macro a `do { ... } while(0)` block to ensure it behaves like a single statement.
 *
 * Thinking:
 *   The requirement is to create a macro that expands to a single statement, preventing
 *   issues when the macro is used in control flow structures (like if-else) or when
 *   it contains multiple statements. A classic way to guarantee single-statement
 *   semantics is to wrap the macro body in a `do { ... } while(0)` loop that executes
 *   only once. The macro will also use temporary variables to evaluate its arguments
 *   only once, avoiding side effects.
 *
 *   For demonstration, I will implement a macro `PRINT_MAX(a, b)` that prints the
 *   maximum of two integer expressions. The macro will be defined as:
 *
 *     #define PRINT_MAX(a, b) do { \\
 *         const int _a = (a); \\
 *         const int _b = (b); \\
 *         printf("Max of %d and %d is %d\\n", _a, _b, (_a > _b) ? _a : _b); \\
 *     } while(0)
 *
 *   The macro uses local constants `_a` and `_b` to store the evaluated arguments,
 *   ensuring each argument is evaluated exactly once. The `do { ... } while(0)`
 *   wrapper guarantees that the macro behaves like a single statement, which is
 *   safe to use in any context, such as:
 *
 *     if (x > 0)
 *         PRINT_MAX(x, y);
 *     else
 *         printf("x is non-positive\n");
 *
 *   The rest of the program demonstrates a simple CUDA vector addition to
 *   provide a complete, runnable .cu file. It allocates two input arrays,
 *   copies them to the GPU, launches a kernel to add them element-wise,
 *   copies the result back, and then prints the maximum of the first
 *   two elements of the result using the macro. Error checking is
 *   performed on CUDA API calls. This example ensures that the
 *   macro works correctly in host code and illustrates its usage.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Macro that prints the maximum of two integer expressions.
// The macro uses a do-while(0) block to ensure it behaves like a single statement.
#define PRINT_MAX(a, b) do { \
    const int _a = (a); \
    const int _b = (b); \
    printf("Max of %d and %d is %d\n", _a, _b, (_a > _b) ? _a : _b); \
} while(0)

// CUDA kernel that adds two integer arrays element-wise.
__global__ void addKernel(const int *a, const int *b, int *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

// Helper macro for CUDA error checking.
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main(void)
{
    const int N = 10;
    const size_t size = N * sizeof(int);

    // Allocate host memory.
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays.
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    // Allocate device memory.
    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    // Copy host data to device.
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel.
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host.
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Print results.
    printf("Result of vector addition:\n");
    for (int i = 0; i < N; ++i)
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);

    // Use the PRINT_MAX macro to display the maximum of the first two elements.
    PRINT_MAX(h_c[0], h_c[1]);

    // Free device memory.
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // Free host memory.
    free(h_a);
    free(h_b);
    free(h_c);

    // Reset device.
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```