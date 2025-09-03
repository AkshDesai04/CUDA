```cpp
/* 
The reduction loop looks like: `for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { 
    if (threadIdx.x < s) { 
        s_data[threadIdx.x] += s_data[threadIdx.x + s]; 
    } 
    __syncthreads(); 
}`

Implement this.

Thinking:
1. We need a kernel that performs a sum reduction over an array of floats.
2. Each block will handle a contiguous chunk of the input array. The block size is a power of two (e.g., 256) to simplify the loop.
3. The kernel loads its chunk into shared memory (`s_data`), then runs the loop described above to reduce the values to a single sum per block.
4. After the loop, thread 0 writes the block's partial sum to an output array (`g_odata`). The host will then sum these partial sums to obtain the final result.
5. For simplicity we assume the input size `n` is a multiple of `blockSize * numBlocks`. If not, we would need bounds checks, but that is omitted here to keep the focus on the reduction loop.
6. Host code allocates input data, copies it to device, launches the kernel, copies the partial sums back, and finally reduces them on the host to print the total sum.
7. We include basic CUDA error checking macros for robustness.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel: block-wise reduction to sum a float array
__global__ void reduce_sum(const float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float s_data[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory; guard against out-of-bounds
    if (idx < n)
        s_data[tid] = g_idata[idx];
    else
        s_data[tid] = 0.0f; // pad with zeros if idx out of range

    __syncthreads();

    // Reduction loop
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        g_odata[blockIdx.x] = s_data[0];
    }
}

int main()
{
    const unsigned int N = 1 << 20;          // Number of elements (1M)
    const unsigned int BLOCK_SIZE = 256;     // Must be power of two
    const unsigned int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate host memory
    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_partial = (float*)malloc(GRID_SIZE * sizeof(float));

    // Initialize input with some values (e.g., all ones)
    for (unsigned int i = 0; i < N; ++i)
        h_in[i] = 1.0f;

    // Allocate device memory
    float *d_in = nullptr;
    float *d_partial = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_partial, GRID_SIZE * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    size_t sharedMemSize = BLOCK_SIZE * sizeof(float);
    reduce_sum<<<GRID_SIZE, BLOCK_SIZE, sharedMemSize>>>(d_in, d_partial, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums back to host
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Final reduction on host
    float totalSum = 0.0f;
    for (unsigned int i = 0; i < GRID_SIZE; ++i)
        totalSum += h_partial[i];

    printf("Total sum: %f\n", totalSum);

    // Clean up
    free(h_in);
    free(h_partial);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));

    return 0;
}
```