/*
Aim: Are atomics supported on shared memory? (Yes). Write a reduction that uses atomics on a shared memory location instead of the synchronized strided reduction.

Thinking:
- The goal is to demonstrate that atomic operations can be performed on shared memory, which is per-block local memory.
- We'll implement a simple sum reduction where each thread adds its portion of the input array to a shared accumulator using atomicAdd.
- Each block will compute a partial sum and write it to a global output array. The host will then sum these partial sums to get the final result.
- Shared memory is allocated dynamically for the kernel launch; we only need a single float per block for the accumulator.
- Initialization: thread 0 sets the shared accumulator to 0.0f; all threads then synchronize before starting the atomicAdd loop.
- Each thread processes elements in a strided manner to cover the entire input.
- After the loop, all threads synchronize again, and thread 0 writes the block's partial sum to global memory.
- The host code sets up data, launches the kernel, copies results back, and aggregates the partial sums.
- This demonstrates atomicAdd on shared memory is allowed and functional.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",             \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    }

/* Kernel that performs block-wise reduction using atomicAdd on shared memory */
__global__ void reduction_atomic(const float* __restrict__ d_in,
                                 float* d_out,
                                 int n)
{
    /* Allocate shared memory: 1 float per block for the accumulator */
    extern __shared__ float s_sum[];

    /* Initialize shared accumulator to 0.0 by thread 0 */
    if (threadIdx.x == 0)
        s_sum[0] = 0.0f;
    __syncthreads();

    /* Stride over input to cover all elements */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
    {
        /* Atomic add to shared accumulator */
        atomicAdd(&s_sum[0], d_in[i]);
    }

    __syncthreads(); /* Ensure all atomic adds complete */

    /* Write block result to global memory */
    if (threadIdx.x == 0)
        d_out[blockIdx.x] = s_sum[0];
}

int main(void)
{
    const int N = 1 << 24; /* 16M elements */
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    /* Allocate host memory */
    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_partial = (float*)malloc(gridSize * sizeof(float));

    /* Initialize input with some values, e.g., 1.0f */
    for (int i = 0; i < N; ++i)
        h_in[i] = 1.0f; /* simple test case */

    /* Allocate device memory */
    float *d_in = nullptr;
    float *d_partial = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_partial, gridSize * sizeof(float)));

    /* Copy input to device */
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Launch kernel: use 1 float of shared memory per block */
    size_t sharedMemSize = sizeof(float); /* one float */
    reduction_atomic<<<gridSize, blockSize, sharedMemSize>>>(d_in, d_partial, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy partial sums back to host */
    CHECK_CUDA(cudaMemcpy(h_partial, d_partial, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    /* Final reduction on host */
    double sum = 0.0;
    for (int i = 0; i < gridSize; ++i)
        sum += h_partial[i];

    printf("Sum of %d elements = %.1f\n", N, sum);

    /* Clean up */
    free(h_in);
    free(h_partial);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_partial));

    return 0;
}
