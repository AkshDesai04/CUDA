/*
Aim: Move the global index calculation and boundary check into a __device__ helper function.

Thinking:
The task is to refactor a typical CUDA kernel so that the computation of the global thread index
and the safety boundary check (ensuring we do not access out-of-range memory) are encapsulated
in a separate __device__ function. This improves readability and reusability, especially in
kernels where the same index logic is used multiple times. To demonstrate this, I will write a
simple vector addition kernel. The __device__ helper will compute the global linear index and
return -1 if the index is out of bounds. The kernel will call this helper and only perform
the addition when a valid index is returned.

The program will:
1. Allocate two input arrays and one output array on the host.
2. Copy the inputs to device memory.
3. Launch a kernel that uses the helper to safely add the vectors.
4. Copy the result back to host and verify it.
5. Clean up all allocated resources.

Error checking macros are included to simplify CUDA API calls.

*/

#include <cuda_runtime.h>
#include <stdio.h>

/* Helper macro for CUDA error checking */
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* __device__ helper that computes global index and performs boundary check */
__device__ int getGlobalIdx(int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        return idx;
    else
        return -1;  // Indicates out-of-bounds
}

/* Kernel that uses the helper function */
__global__ void vectorAdd(const float *a, const float *b, float *c, int N)
{
    int idx = getGlobalIdx(N);
    if (idx >= 0)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;           // 1M elements
    const size_t size = N * sizeof(float);

    /* Host allocations */
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Device allocations */
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void **)&d_a, size));
    CHECK_CUDA(cudaMalloc((void **)&d_b, size));
    CHECK_CUDA(cudaMalloc((void **)&d_c, size));

    /* Copy data to device */
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    bool success = true;
    for (int i = 0; i < 10; ++i)
    {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected)
        {
            printf("Mismatch at index %d: host %f, device %f\n", i, expected, h_c[i]);
            success = false;
            break;
        }
    }
    if (success)
        printf("Vector addition successful. Sample results verified.\n");

    /* Cleanup */
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}
