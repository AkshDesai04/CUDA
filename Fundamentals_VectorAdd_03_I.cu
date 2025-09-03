/*
Move the global index calculation and boundary check into a `__device__` helper function.
--------------------------------------------------------------------
Thinking:

The goal is to extract the logic that determines the global thread index from
the kernel into a reusable device function.  This has two advantages:

1. It reduces duplication if multiple kernels need the same index logic.
2. It keeps kernels clean and easier to read.

For a 1‑D kernel the global index is simply

    idx = blockIdx.x * blockDim.x + threadIdx.x;

The boundary check is usually `if (idx >= N) return;`.  We encapsulate both
steps in a single device function that returns the computed index or
`-1` to indicate an out‑of‑bounds thread.  The kernel then checks the
return value and exits early if appropriate.

The example below implements a vector addition (`c = a + b`) using this
helper.  It includes a small host program that allocates memory,
initializes data, launches the kernel, and verifies the result.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Device helper that computes the global index and performs the boundary check.
// If the thread is out of bounds, it returns -1.
__device__ int globalIdxAndCheck(int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return -1;        // Signal out-of-bounds
    return idx;
}

// Simple vector addition kernel using the helper function.
__global__ void addKernel(const float *a, const float *b, float *c, int N)
{
    int idx = globalIdxAndCheck(N);
    if (idx == -1)  // Out of bounds, exit early
        return;

    c[idx] = a[idx] + b[idx];
}

int main(void)
{
    const int N = 1 << 20;            // 1M elements
    const size_t size = N * sizeof(float);

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
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify result (simple check on a few elements)
    bool success = true;
    for (int i = 0; i < 10 && success; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n",
                    i, h_c[i], expected);
            success = false;
        }
    }
    if (success)
        printf("Vector addition succeeded.\n");
    else
        printf("Vector addition failed.\n");

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
