/*
The first kernel will take the large input array and an output array sized to the number of blocks. Each block writes its partial sum to d_partial_sums[blockIdx.x].

Thinking:
To implement the described kernel I will create a CUDA program that computes the sum of a large float array using a two-stage reduction:
1. The first kernel (`partialSum`) will process the input array in parallel, with each block handling a contiguous chunk of data. Inside each block, threads cooperate via shared memory to reduce the chunk to a single value, which is then written to a per‑block partial sums array (`d_partial_sums[blockIdx.x]`). The kernel will use a classic parallel reduction pattern: each thread loads one or more elements into a shared array, then iteratively halving the number of active threads until one thread holds the block’s total.
2. After launching the first kernel, the host will copy the `d_partial_sums` array back to the CPU and perform the final summation on the CPU (for simplicity). This demonstrates that the partial sums from each block have been correctly produced by the GPU.
The program includes basic error checking, allocation of host and device memory, initialization of the input array with known values (e.g., sequential numbers), kernel launch parameters, and a final verification that the computed sum matches the expected analytical result.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/* CUDA error checking macro */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",           \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Kernel that computes partial sums for each block */
__global__ void partialSum(const float *input, float *partial_sums, int N)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x; // 2 elements per thread

    /* Load data into shared memory, each thread loads two elements if possible */
    float sum = 0.0f;
    if (idx < N)
        sum = input[idx];
    if (idx + blockDim.x < N)
        sum += input[idx + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    /* Parallel reduction in shared memory */
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    /* Write block's partial sum to output */
    if (tid == 0)
        partial_sums[blockIdx.x] = sdata[0];
}

/* Helper function to compute expected sum analytically */
double expectedSum(int N)
{
    /* Sum of numbers 0,1,...,N-1 is N*(N-1)/2 */
    return (double)N * (N - 1) / 2.0;
}

int main(void)
{
    /* Problem size */
    const int N = 1 << 26;          /* ~67 million elements */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    /* Host memory allocation */
    float *h_input = (float *)malloc(N * sizeof(float));
    if (!h_input) {
        fprintf(stderr, "Failed to allocate host input array\n");
        return EXIT_FAILURE;
    }

    /* Initialize input array with 0,1,2,...,N-1 */
    for (int i = 0; i < N; ++i)
        h_input[i] = (float)i;

    /* Device memory allocation */
    float *d_input = NULL;
    float *d_partial_sums = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_partial_sums, blocksPerGrid * sizeof(float)));

    /* Copy input data to device */
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Launch kernel */
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    partialSum<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_partial_sums, N);
    CUDA_CHECK(cudaGetLastError());

    /* Copy partial sums back to host */
    float *h_partial_sums = (float *)malloc(blocksPerGrid * sizeof(float));
    if (!h_partial_sums) {
        fprintf(stderr, "Failed to allocate host partial sums array\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    /* Compute final sum on host */
    double gpu_sum = 0.0;
    for (int i = 0; i < blocksPerGrid; ++i)
        gpu_sum += (double)h_partial_sums[i];

    /* Compute expected sum */
    double ref_sum = expectedSum(N);

    /* Verify result */
    printf("GPU sum = %.0f, Expected sum = %.0f\n", gpu_sum, ref_sum);
    if (fabs(gpu_sum - ref_sum) < 1e-3)
        printf("Result is correct.\n");
    else
        printf("Result is incorrect!\n");

    /* Clean up */
    free(h_input);
    free(h_partial_sums);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial_sums));

    return 0;
}
