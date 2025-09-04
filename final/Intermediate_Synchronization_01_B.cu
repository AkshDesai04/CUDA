/*
 The second kernel is launched with just a single block. This block loads the d_partial_sums array into shared memory and performs a final reduction on it.

 In this program we implement a two‑stage parallel reduction of an array of floating‑point numbers using CUDA. The first kernel performs a block‑wise reduction: each block processes a chunk of the input array, loads the data into shared memory, and reduces it to a single partial sum that is written to a global partial_sums array. We use a standard two‑step reduction pattern: each thread loads up to two elements (to keep the memory bandwidth high) and then iteratively halves the number of active threads, summing the values in shared memory until only one value remains per block.

 The second kernel performs the final reduction of the partial_sums array. According to the prompt we launch this kernel with a single block. All the partial sums from the first stage are loaded into shared memory and reduced in the same way as the first kernel, but this time there is only one block to operate on the whole partial_sums array. The final result is written back to host memory.

 The program sets up a simple test: it creates a large array of floats, fills it with a constant value, runs the reduction kernels, and prints the result. We also perform a simple CPU sum for verification. Error checking for CUDA API calls is performed using a helper macro. The code is self‑contained and can be compiled with nvcc (e.g., nvcc -arch=sm_70 -o reduce reduce.cu).
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

/* Kernel 1: block‑wise reduction. Each thread loads up to two elements from the input array. */
__global__ void reduce_kernel(const float *input, float *partial_sums, size_t N)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;

    if (idx < N)
        sum += input[idx];
    if (idx + blockDim.x < N)
        sum += input[idx + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = sdata[0];
}

/* Kernel 2: final reduction over partial_sums. Launched with a single block. */
__global__ void final_reduce_kernel(const float *partial_sums, float *output, size_t N)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;

    // Load partial sums into shared memory
    if (tid < N)
        sdata[tid] = partial_sums[tid];
    else
        sdata[tid] = 0.0f;   // pad with zero if N is not a multiple of blockDim.x

    __syncthreads();

    // Reduce within the single block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        *output = sdata[0];
}

int main()
{
    const size_t N = 1 << 20;          // 1M elements
    const size_t bytes = N * sizeof(float);

    float *h_input = (float *)malloc(bytes);
    if (!h_input) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input array
    for (size_t i = 0; i < N; ++i)
        h_input[i] = 1.0f;   // all ones; expected sum = N

    float *d_input = nullptr;
    float *d_partial_sums = nullptr;
    float *d_output = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_input, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_partial_sums, (N + 1) * sizeof(float))); // allocate max possible
    CHECK_CUDA(cudaMalloc((void **)&d_output, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Determine block size and grid size
    const unsigned int BLOCK_SIZE = 256;
    unsigned int gridSize = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    size_t sharedMemSize = BLOCK_SIZE * sizeof(float);

    // Launch first kernel
    reduce_kernel<<<gridSize, BLOCK_SIZE, sharedMemSize>>>(d_input, d_partial_sums, N);
    CHECK_CUDA(cudaGetLastError());

    // Final reduction: one block
    // We launch with blockDim.x = gridSize (number of partial sums)
    // Ensure blockDim.x <= 1024 (max threads per block)
    unsigned int finalBlockSize = gridSize;
    if (finalBlockSize > 1024) {
        // If too many partial sums, we could do a second level reduction.
        // For simplicity, we assume N is not extremely large.
        fprintf(stderr, "Too many partial sums for a single block. Exiting.\n");
        return EXIT_FAILURE;
    }

    sharedMemSize = finalBlockSize * sizeof(float);
    final_reduce_kernel<<<1, finalBlockSize, sharedMemSize>>>(d_partial_sums, d_output, gridSize);
    CHECK_CUDA(cudaGetLastError());

    float h_output = 0.0f;
    CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // Compute CPU sum for verification
    double cpu_sum = 0.0;
    for (size_t i = 0; i < N; ++i)
        cpu_sum += h_input[i];

    printf("GPU sum:   %f\n", h_output);
    printf("CPU sum:   %f\n", cpu_sum);

    // Clean up
    free(h_input);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_partial_sums));
    CHECK_CUDA(cudaFree(d_output));

    return EXIT_SUCCESS;
}
