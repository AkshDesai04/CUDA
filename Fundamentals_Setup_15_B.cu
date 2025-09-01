```c
/*
Aim: Call `cudaDeviceReset()` at the very beginning of your `main` function to ensure a clean state.

Thinking:
- The program will demonstrate a minimal CUDA example (vector addition) while adhering to the requirement of calling `cudaDeviceReset()` right at the start of `main`.
- To keep the example self‑contained, I will:
  1. Include the necessary headers (`<stdio.h>` and `<cuda_runtime.h>`).
  2. Define a simple kernel `add` that adds two integer arrays.
  3. Use a macro `CUDA_CHECK` for concise error handling of CUDA API calls.
  4. Allocate and initialize host vectors, copy them to device memory, launch the kernel, copy the result back, and verify correctness.
  5. Call `cudaDeviceReset()` only at the very beginning of `main` as specified, and optionally at the end to clean up resources.
- The code will compile as a standard CUDA C file (`.cu`) and can be built with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

__global__ void add(const int *a, const int *b, int *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main(void)
{
    /* Reset the device to ensure a clean state before any CUDA calls. */
    CUDA_CHECK(cudaDeviceReset());

    const int N = 1 << 20;  // 1 million elements
    size_t bytes = N * sizeof(int);

    int *h_a = (int *)malloc(bytes);
    int *h_b = (int *)malloc(bytes);
    int *h_c = (int *)malloc(bytes);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    /* Verify results */
    int correct = 1;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            correct = 0;
            printf("Mismatch at index %d: %d + %d != %d\n",
                   i, h_a[i], h_b[i], h_c[i]);
            break;
        }
    }
    if (correct)
        printf("Vector addition succeeded.\n");
    else
        printf("Vector addition failed.\n");

    /* Clean up */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    /* Optional: reset device again at the end */
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```