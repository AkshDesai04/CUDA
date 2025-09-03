```cuda
/*
Aim: Implement a parallel maximum-finding algorithm using the same two-pass reduction strategy as the sum reduction.

Thinking Process:
1. The problem is essentially a reduction of an array to a single maximum value. A two-pass strategy
   is used: first, each block reduces a portion of the array to a partial maximum; second, those
   partial results are reduced again until a single value remains.
2. We use CUDA's parallelism: each thread reads one element (or INT_MIN if out of bounds),
   places it in shared memory, then performs a tree‑like reduction within the block.
3. For the first pass, we launch one block per chunk of BLOCK_SIZE elements. The block writes its
   maximum to a global array `partial`.
4. For the second pass (and subsequent passes if necessary), we treat the `partial` array as the
   new input and reuse the same kernel. The kernel arguments allow us to pass the same array
   as both input and output when the size shrinks, which keeps the code simple.
5. The host code loops, launching the kernel repeatedly until only one element remains in the
   partial array. This value is then copied back to host memory.
6. Edge cases:
   - If the input length is not a multiple of BLOCK_SIZE, the kernel must handle out‑of‑range
     indices by using INT_MIN (the minimum integer) as the neutral element for max.
   - The final reduction may finish with fewer than BLOCK_SIZE threads; the kernel handles this
     naturally due to the bounds check.
7. Error checking is performed via a helper macro to ensure CUDA calls succeed.
8. The program includes a small test harness: it fills an array with random integers, copies it
   to the GPU, runs the reduction, and prints the maximum found along with the time taken.
9. Compile instructions are provided in a comment for convenience.

The following code implements all of this in a single .cu file. Compile with:
   nvcc -o max_reduction max_reduction.cu
And run:
   ./max_reduction
*/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

/* Helper macro for CUDA error checking */
#define checkCuda(call)                                                     \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",            \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Kernel: reduce a chunk of input array to its maximum value */
__global__ void reduceMaxKernel(const int *input, int *output, int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load element or INT_MIN if out of bounds */
    sdata[tid] = (gid < n) ? input[gid] : INT_MIN;
    __syncthreads();

    /* In‑block reduction */
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int other = sdata[tid + stride];
            if (other > sdata[tid]) sdata[tid] = other;
        }
        __syncthreads();
    }

    /* Write block result */
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/* Host function to perform the two‑pass reduction */
int parallelMax(const int *d_input, int n)
{
    int *d_partial = NULL;
    int *d_result = NULL;

    /* First pass: compute partial maxima */
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    checkCuda(cudaMalloc((void**)&d_partial, blocks * sizeof(int)));

    reduceMaxKernel<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
        d_input, d_partial, n);
    checkCuda(cudaGetLastError());

    /* Subsequent passes: reduce the partial results until one value remains */
    int currentSize = blocks;
    while (currentSize > 1) {
        int nextBlocks = (currentSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        /* Launch kernel: input and output are the same array to reuse allocation */
        reduceMaxKernel<<<nextBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
            d_partial, d_partial, currentSize);
        checkCuda(cudaGetLastError());
        currentSize = nextBlocks;
    }

    /* Copy final result back to host */
    int maxVal;
    checkCuda(cudaMemcpy(&maxVal, d_partial, sizeof(int), cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(d_partial));
    return maxVal;
}

int main()
{
    const int N = 1 << 20;  // 1M elements
    int *h_data = (int*)malloc(N * sizeof(int));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize data with random integers */
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_data[i] = rand() % 1000000;  // random values between 0 and 999999
    }

    /* Allocate device memory and copy data */
    int *d_data = NULL;
    checkCuda(cudaMalloc((void**)&d_data, N * sizeof(int)));
    checkCuda(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    /* Timing the reduction */
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    checkCuda(cudaEventRecord(start));

    int maxVal = parallelMax(d_data, N);

    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));

    float ms;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));

    printf("Maximum value: %d\n", maxVal);
    printf("Time taken: %.3f ms\n", ms);

    /* Clean up */
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    checkCuda(cudaFree(d_data));
    free(h_data);

    return EXIT_SUCCESS;
}
```