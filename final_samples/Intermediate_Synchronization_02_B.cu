```cuda
/* 
Aim: Implement the two-kernel approach for finding the maximum value.

Thinking:
The task requires a CUDA program that uses two distinct kernel launches to perform
a reduction that finds the maximum value in an array. The typical two‑kernel
strategy is:

1. **Block‑level reduction** – Each block processes a chunk of the input array
   and writes a single maximum value (the block’s local maximum) to a temporary
   array (`d_partial`). This is achieved by loading data into shared memory,
   performing a classic binary reduction within the block, and storing the
   result in `d_partial[blockIdx.x]`.

2. **Global reduction** – The temporary array of block maxima is further
   reduced to a single value. Since the number of block maxima can still be
   larger than the number of threads in a single block, we use a second kernel
   that executes on **one block only**. Each thread scans a portion of the
   `d_partial` array, keeps a local maximum, writes that local maximum into
   shared memory, and then performs a final reduction across the block to
   produce the global maximum.

The main program:
- Allocates and initializes a large array of random integers on the host.
- Copies the data to device memory.
- Launches the block‑reduction kernel.
- Launches the single‑block reduction kernel.
- Copies the final maximum back to host and prints it.

We include error checking for CUDA calls, use a fixed thread block size of
256 (common for reductions), and guard all memory accesses with bounds checks
to handle arrays that are not multiples of the block size. The program is
self‑contained and can be compiled with `nvcc`. 
*/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// Kernel 1: Block-level reduction to compute partial maxima
__global__ void blockMaxReduce(const int *d_in, int *d_out, int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory with bounds checking
    if (idx < n)
        sdata[tid] = d_in[idx];
    else
        sdata[tid] = INT_MIN;  // Sentinel for unused elements

    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid])
                sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write block's maximum to output array
    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

// Kernel 2: Single‑block reduction to compute global maximum from partials
__global__ void finalMaxReduce(const int *d_partial, int *d_result, int num_partials)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = tid;

    // Each thread scans a chunk of the partial array
    int local_max = INT_MIN;
    for (int i = idx; i < num_partials; i += blockDim.x) {
        int val = d_partial[i];
        if (val > local_max)
            local_max = val;
    }

    // Store local maximum into shared memory
    sdata[tid] = local_max;
    __syncthreads();

    // Reduce shared memory to obtain global maximum
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid])
                sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the global maximum to output
    if (tid == 0)
        *d_result = sdata[0];
}

int main()
{
    const int N = 1 << 20;               // Size of the input array (1M elements)
    const int threadsPerBlock = 256;     // Threads per block for block reduction

    // Allocate host memory
    int *h_in = (int*)malloc(N * sizeof(int));
    if (!h_in) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with random integers
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_in[i] = rand();
    }

    // Device memory pointers
    int *d_in = nullptr;
    int *d_partial = nullptr;
    int *d_result = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(int)));

    // Calculate number of blocks for the first kernel
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate partial results array on device
    CUDA_CHECK(cudaMalloc((void**)&d_partial, numBlocks * sizeof(int)));

    // Allocate result array on device
    CUDA_CHECK(cudaMalloc((void**)&d_result, sizeof(int)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch block-level reduction kernel
    size_t sharedMemSize = threadsPerBlock * sizeof(int);
    blockMaxReduce<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_in, d_partial, N);
    CUDA_CHECK(cudaGetLastError());

    // Launch final reduction kernel (single block)
    int threadsFinal = 256;  // Should be enough for most partial sizes
    sharedMemSize = threadsFinal * sizeof(int);
    finalMaxReduce<<<1, threadsFinal, sharedMemSize>>>(d_partial, d_result, numBlocks);
    CUDA_CHECK(cudaGetLastError());

    // Copy the final maximum back to host
    int h_max = 0;
    CUDA_CHECK(cudaMemcpy(&h_max, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    // Find the maximum on host for verification
    int host_max = h_in[0];
    for (int i = 1; i < N; ++i) {
        if (h_in[i] > host_max)
            host_max = h_in[i];
    }

    // Print results
    printf("GPU maximum: %d\n", h_max);
    printf("CPU maximum: %d\n", host_max);
    printf("Verification: %s\n", (h_max == host_max) ? "PASS" : "FAIL");

    // Clean up
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_result));
    free(h_in);

    return EXIT_SUCCESS;
}
```