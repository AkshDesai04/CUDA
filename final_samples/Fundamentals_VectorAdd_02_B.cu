/*
Make the vector size `N` and the block size `BLOCK_SIZE` configurable (e.g., using `const int` or `#define`). Calculate the grid size dynamically.

In this solution we create a simple CUDA program that performs element‑wise addition of two vectors. The key points that satisfy the user’s request are:

1. **Configurability** – The vector size `N` and the block size `BLOCK_SIZE` are declared as `const int` so that they can be changed by editing a single line. Using `#define` would also work, but constants are type‑safe.
2. **Dynamic grid sizing** – The grid size is computed at runtime using the classic ceiling division formula `(N + BLOCK_SIZE - 1) / BLOCK_SIZE`. This ensures that all elements are processed even when `N` is not a multiple of `BLOCK_SIZE`.
3. **Self‑contained example** – The program allocates host memory, initializes the input vectors, copies data to the device, launches the kernel, copies the result back, and prints a few values for verification.
4. **Error checking** – Basic CUDA error checks are included to aid debugging.

With these design choices the program demonstrates the requested configurability and dynamic grid calculation while remaining clear and concise. */

#include <stdio.h>
#include <cuda_runtime.h>

/* User configurable parameters */
const int N = 1024;          // Size of the vectors
const int BLOCK_SIZE = 256;  // Threads per block

/* CUDA kernel for vector addition */
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                               \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),    \
                    cudaGetErrorString(err));                         \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main(void)
{
    /* Allocate host memory */
    size_t bytes = N * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Determine grid size dynamically */
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE);
    dim3 grid(gridSize);

    /* Launch kernel */
    vectorAdd<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    printf("Sample results (index : A + B = C):\n");
    for (int i = 0; i < 5 && i < N; ++i)
        printf("  %d : %f + %f = %f\n", i, h_A[i], h_B[i], h_C[i]);

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
