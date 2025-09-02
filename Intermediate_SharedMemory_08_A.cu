/*
The reduction loop looks like: `for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) { s_data[threadIdx.x] += s_data[threadIdx.x + s]; } __syncthreads(); }` Implement this.

Thinking process:
1. The user wants a CUDA program that implements the given reduction loop. 
2. I will create a simple sum reduction kernel that uses the provided loop structure.
3. The kernel will load two elements per thread into shared memory to maximize coalescing and then perform the reduction using the loop.
4. After the kernel finishes, the host will gather the per‑block partial sums and compute the final result on the CPU for simplicity.
5. Basic error handling and resource cleanup will be included.
6. The code is written as a single .cu file, with a multi‑line comment at the top containing the exact aim and my explanation.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel performing block‑level reduction using the specified loop
__global__ void reduce_kernel(const float *in, float *out, int N)
{
    extern __shared__ float s_data[];

    unsigned int tid = threadIdx.x;
    // Each thread loads two elements if available
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (idx < N)          sum += in[idx];
    if (idx + blockDim.x < N) sum += in[idx + blockDim.x];

    s_data[tid] = sum;
    __syncthreads();

    // Reduction loop (exactly as requested)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // Write block's result to global memory
    if (tid == 0)
        out[blockIdx.x] = s_data[0];
}

int main(void)
{
    const int N = 1 << 20;                // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory and initialize
    float *h_in = (float*)malloc(size);
    for (int i = 0; i < N; ++i)
        h_in[i] = 1.0f;                   // simple test data

    // Allocate device memory
    float *d_in, *d_partial;
    cudaMalloc((void**)&d_in, size);

    // Determine block and grid sizes
    const int blockSize = 256;
    const int elemsPerBlock = blockSize * 2;
    const int gridSize = (N + elemsPerBlock - 1) / elemsPerBlock;

    cudaMalloc((void**)&d_partial, gridSize * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Launch reduction kernel
    reduce_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_in, d_partial, N);

    // Copy partial results back to host
    float *h_partial = (float*)malloc(gridSize * sizeof(float));
    cudaMemcpy(h_partial, d_partial, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on host
    float total = 0.0f;
    for (int i = 0; i < gridSize; ++i)
        total += h_partial[i];

    printf("Total sum = %f\n", total);

    // Clean up
    cudaFree(d_in);
    cudaFree(d_partial);
    free(h_in);
    free(h_partial);

    return 0;
}
