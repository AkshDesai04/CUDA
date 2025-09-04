```cuda
/* 
Aim of the program: 
"Generalize this to require more than two passes if the number of partial sums is still too large for a single block."

Thinking: 
The original reduction pattern used two passes – first compute partial sums per block, then sum those partials in a second kernel.  
If the array is extremely large, the number of partial sums may still exceed what a single block can handle. 
To address this, we iteratively launch the reduction kernel until the number of elements to sum drops to one. 
Each iteration reduces the array size by roughly a factor of 2*blockDim.  
We keep two device buffers (ping‑pong) and swap them each pass.  
The kernel uses shared memory for intra‑block reduction and handles the case where the array length is not a multiple of the block size.  
Finally, after the loop terminates, we copy the single remaining element back to host and print it. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Kernel for block-wise reduction (sum)
__global__ void reduceKernel(const float* g_in, float* g_out, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;

    if (i < n)               sum += g_in[i];
    if (i + blockDim.x < n)  sum += g_in[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // Intra-block reduction
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write block's partial sum to global memory
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

int main(int argc, char** argv) {
    // Example: reduce an array of size N
    size_t N = 1 << 20; // 1,048,576 elements (can be changed)
    float* h_in = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; ++i) h_in[i] = 1.0f; // simple data

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, ((N + 511) / 512) * sizeof(float))); // enough for first pass

    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    size_t currentSize = N;
    float* src = d_in;
    float* dst = d_out;

    while (currentSize > 1) {
        int blocks = (int)((currentSize + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2));
        reduceKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(src, dst, currentSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Prepare for next pass
        currentSize = blocks;
        // swap src and dst
        float* temp = src;
        src = dst;
        dst = temp;
    }

    // Copy the final sum back to host
    float h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, src, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Sum of %zu elements = %f\n", N, h_result);

    // Clean up
    free(h_in);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
```