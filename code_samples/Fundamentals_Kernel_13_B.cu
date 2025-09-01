/*
Aim of the program: 
Write a kernel that takes three pointers, `a`, `b`, and `c`, and computes `c[i] = a[i] + b[i]`.

Thinking:
1. The core of the program is a CUDA kernel that adds two arrays element‑wise.
2. The kernel must be written as an `__global__` function so that it can be launched from the host.
3. Each thread will handle one element: the index is computed from block and thread indices.
4. To make the kernel usable with arbitrary array sizes, we pass the length `n` and guard against out‑of‑bounds accesses.
5. The host code will:
   - Allocate input and output arrays on the host.
   - Allocate device memory for the three arrays.
   - Copy inputs to device.
   - Launch the kernel with enough blocks to cover `n` elements.
   - Copy the result back to the host.
   - Verify and print a few results.
   - Clean up all allocated memory.
6. Error checking will be added after each CUDA API call to aid debugging.
7. The program uses `float` for simplicity, but the kernel could be templated or changed to `int` if needed.
8. The code is fully self‑contained and can be compiled with `nvcc`:
   ```
   nvcc -o vector_add vector_add.cu
   ./vector_add
   ```
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that adds two vectors element‑wise
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA(cudaMalloc((void**)&d_c, size));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Verification of first 10 elements:\n");
    for (int i = 0; i < 10; ++i) {
        printf("c[%d] = %f\n", i, h_c[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
