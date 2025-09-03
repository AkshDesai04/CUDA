```cpp
/*
Aim: Handle large vectors with boundary checks.

Thinking:
The requirement is to demonstrate a CUDA program that operates on large vectors while ensuring
that the kernel performs proper boundary checks to avoid out‑of‑bounds memory accesses.  I chose
a simple element‑wise vector addition as the workload because it is easy to reason about and
illustrates the boundary check clearly.  The kernel calculates a global index and then
checks whether that index is less than the vector size `N` before accessing the arrays.
On the host side, I allocate and initialise two input vectors, copy them to device memory,
launch the kernel with a suitable grid and block configuration, copy the result back, and
perform a quick verification of the first few elements.  CUDA error checking macros are
included to surface any launch or memory copy failures.  The code accepts an optional
command‑line argument for the vector size; otherwise it defaults to a size that is
substantial but still fits comfortably in GPU memory on most systems.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                    \
        }                                                           \
    } while (0)

__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char *argv[])
{
    size_t N = 0;
    if (argc > 1) {
        N = strtoull(argv[1], NULL, 10);
    } else {
        N = 1 << 26; // default 67,108,864 elements (~256 MB per array)
    }
    printf("Vector size: %zu\n", N);

    /* Allocate host memory */
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));

    /* Initialize input vectors */
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_B, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_C, N * sizeof(float)));

    /* Copy inputs to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Kernel launch configuration */
    int threadsPerBlock = 256;
    int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);

    /* Launch kernel */
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Verify a few elements */
    bool success = true;
    for (size_t i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            fprintf(stderr, "Mismatch at index %zu: got %f, expected %f\n",
                    i, h_C[i], expected);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Verification passed for first 10 elements.\n");
    }

    /* Free resources */
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```