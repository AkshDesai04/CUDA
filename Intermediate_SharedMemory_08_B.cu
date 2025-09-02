```cuda
/*
Aim of the program:
Implement a max-finding reduction using the same pattern.

Thinking:
To compute the maximum value of an array on the GPU we use a two-stage
parallel reduction that follows the classic reduction pattern:
1. Each block loads a segment of the input array into shared memory.
2. Within the block the threads perform a binary tree reduction
   to find the maximum of that segment.
3. The block writes its local maximum to an output array.
4. We repeat the reduction on the output array until a single
   maximum remains.

The kernel uses an unrolled reduction over pairs of elements
(`idx` and `idx + blockDim.x`) to reduce divergence and improve
coalesced memory accesses. We also use the `extern __shared__`
array for efficient intra-block communication. The host code
allocates the input array, copies it to the device, performs
multiple kernel launches until the array size is 1, then copies
the result back to the host.

The program is written in plain C with CUDA extensions and is
self‑contained in a single .cu file. No external libraries are
required. Compile with `nvcc -arch=sm_52 -o max_reduce max_reduce.cu`
and run with `./max_reduce`.

The maximum value will be printed to standard output.
*/

#include <stdio.h>
#include <float.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 256

// Kernel to reduce a segment of the array to its maximum.
// Each block processes up to 2*blockDim.x elements.
__global__ void reduceMaxKernel(const float *in, float *out, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load two elements per thread into shared memory, if available.
    float val = -FLT_MAX;
    if (idx < n)
        val = in[idx];
    if (idx + blockDim.x < n) {
        float tmp = in[idx + blockDim.x];
        if (tmp > val) val = tmp;
    }
    sdata[tid] = val;
    __syncthreads();

    // Intra-block reduction: binary tree.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block's local maximum to the output array.
    if (tid == 0)
        out[blockIdx.x] = sdata[0];
}

// Helper function to launch the kernel and perform iterative reductions.
float findMaxOnDevice(const float *d_in, int size)
{
    int currSize = size;
    const float *d_curr = d_in;
    float *d_next = nullptr;
    cudaMalloc(&d_next, (currSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * sizeof(float));

    while (currSize > 1) {
        int blocks = (currSize + (THREADS_PER_BLOCK * 2 - 1)) / (THREADS_PER_BLOCK * 2);
        int sharedMemSize = THREADS_PER_BLOCK * sizeof(float);

        reduceMaxKernel<<<blocks, THREADS_PER_BLOCK, sharedMemSize>>>(d_curr, d_next, currSize);
        cudaDeviceSynchronize();

        // Prepare for next iteration
        currSize = blocks;
        cudaFree((void*)d_curr);
        d_curr = d_next;
        if (currSize > 1) {
            cudaMalloc(&d_next, (currSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * sizeof(float));
        }
    }

    // Copy final result back to host
    float result;
    cudaMemcpy(&result, d_curr, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree((void*)d_curr);
    return result;
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    // Allocate and initialize host memory
    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_data[i] = (float)(rand()) / RAND_MAX * 1000.0f; // Random values 0-1000
    }

    // Allocate device memory
    float *d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // Find maximum on GPU
    float maxVal = findMaxOnDevice(d_data, N);
    printf("Maximum value found: %f\n", maxVal);

    // Cleanup
    cudaFree(d_data);
    free(h_data);

    return 0;
}
```