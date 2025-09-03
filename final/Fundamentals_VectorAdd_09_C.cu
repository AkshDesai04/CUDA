```cpp
/*
Aim of the program:
Implement `D[i] = (A[i] > B[i]) ? A[i] * s1 : B[i] * s2`, where `s1` and `s2` are scalars.

Thinking process:
1. The problem requires a CUDA kernel that performs an element‑wise comparison between two arrays A and B and
   computes the result array D according to the given ternary expression.
2. We will use single‑precision floating point (`float`) for simplicity, but the kernel can be easily adapted
   to other types by changing the template parameter.
3. The kernel will be launched with a grid of blocks and threads such that each thread handles one element
   `i` if `i < n`. Boundary checking (`i < n`) ensures safe access when the total number of threads
   exceeds `n`.
4. In `main()` we will:
   - Define a problem size `N` (e.g., 1<<20 elements).
   - Allocate host arrays for A, B, and D.
   - Initialize A and B with deterministic values so that results are reproducible.
   - Allocate device memory for A, B, and D.
   - Copy host data to device.
   - Launch the kernel with appropriate block and grid dimensions.
   - Copy the result back to the host.
   - Optionally print a few elements of the output to verify correctness.
   - Clean up allocated memory.
5. Error handling: we will wrap CUDA API calls in a macro `CHECK_CUDA_ERROR` that prints the error
   and exits if any call fails.
6. The program is self‑contained, uses standard C++/C features, and compiles to a single `.cu` file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CHECK_CUDA_ERROR(call)                                            \
    do {                                                                   \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",             \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// CUDA kernel performing the element-wise operation
__global__ void compute(const float *A, const float *B, float *D,
                        float s1, float s2, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        D[i] = (A[i] > B[i]) ? (A[i] * s1) : (B[i] * s2);
    }
}

int main(void)
{
    const int N = 1 << 20; // Number of elements (1,048,576)
    const size_t bytes = N * sizeof(float);

    // Host allocations
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_D = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_D)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with some pattern
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)(i % 1000) * 0.001f;      // Values in [0, 0.999]
        h_B[i] = (float)(N - i) * 0.0005f;        // Values decreasing from ~0.5
    }

    // Scalars
    const float s1 = 2.5f;
    const float s2 = -1.3f;

    // Device allocations
    float *d_A, *d_B, *d_D;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_D, bytes));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    compute<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, s1, s2, N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    for (int i = 0; i < 10; ++i)
    {
        printf("D[%d] = %f  (A=%f, B=%f)\n", i, h_D[i], h_A[i], h_B[i]);
    }

    // Clean up
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_D));
    free(h_A);
    free(h_B);
    free(h_D);

    return EXIT_SUCCESS;
}
```