```cuda
/*
Aim: For a 3-point stencil `output[i] = w_l*in[i-1] + w_c*in[i] + w_r*in[i+1]`, each thread needs its neighbors. If a block loads N elements for N threads, it must actually load N+2 elements into shared memory.

Thinking:
- In CUDA, each block can load a chunk of the input array into shared memory to reduce global memory traffic. 
- For a 3‑point stencil, each thread requires its left and right neighbor.  
- If a block has `blockDim.x` threads, each thread will compute one output element.  
- To provide the needed neighbors for every thread in the block, we need to load an extra element at the left edge (index `-1` relative to the block) and an extra element at the right edge (index `+1`).  
- Thus, the shared array size is `blockDim.x + 2`.  
- Thread `0` loads the left halo element, thread `blockDim.x-1` loads the right halo element.  
- All threads load their own element into the shared array at index `threadIdx.x + 1`.  
- Synchronize the block before performing the stencil calculation.  
- Handle boundary conditions by setting out‑of‑range accesses to zero.  
- The kernel receives the weights as arguments so it can be reused with different values.  
- In the host code we allocate memory, initialize data, launch the kernel with a grid that covers the entire array, copy back, and perform a simple verification.  
- We use CUDA events to time the kernel execution.  
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel performing a 3‑point stencil using shared memory
__global__ void stencil3pt(const float* __restrict__ in,
                           float* __restrict__ out,
                           int N,
                           float w_l, float w_c, float w_r)
{
    // Allocate shared memory for blockDim.x + 2 elements
    extern __shared__ float s[];

    int t = threadIdx.x;
    int g = blockIdx.x * blockDim.x + t;

    // Load the element belonging to this thread
    if (g < N)
        s[t + 1] = in[g];
    else
        s[t + 1] = 0.0f; // Out of bounds, treat as zero

    // Load left halo (only thread 0 does this)
    if (t == 0)
    {
        int leftIdx = g - 1;
        s[0] = (leftIdx >= 0) ? in[leftIdx] : 0.0f;
    }

    // Load right halo (only the last thread does this)
    if (t == blockDim.x - 1)
    {
        int rightIdx = g + 1;
        s[blockDim.x + 1] = (rightIdx < N) ? in[rightIdx] : 0.0f;
    }

    __syncthreads();

    // Compute the stencil result
    if (g < N)
    {
        out[g] = w_l * s[t] + w_c * s[t + 1] + w_r * s[t + 2];
    }
}

int main(void)
{
    const int N = 1 << 20;          // Size of the array (e.g., 1M elements)
    const int blockSize = 256;      // Threads per block
    const float w_l = 0.25f;
    const float w_c = 0.5f;
    const float w_r = 0.25f;

    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    if (!h_in || !h_out) { fprintf(stderr, "Failed to allocate host memory\n"); return 1; }

    // Initialize input array with some pattern
    for (int i = 0; i < N; ++i)
        h_in[i] = (float)(i % 100) / 100.0f;

    // Allocate device memory
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);

    // Copy input to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Determine grid dimensions
    int gridSize = (N + blockSize - 1) / blockSize;
    size_t sharedMemSize = (blockSize + 2) * sizeof(float);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start
    cudaEventRecord(start, 0);

    // Launch kernel
    stencil3pt<<<gridSize, blockSize, sharedMemSize>>>(d_in, d_out, N, w_l, w_c, w_r);

    // Record stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %f ms\n", ms);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Simple verification: print first 10 results
    printf("First 10 output values:\n");
    for (int i = 0; i < 10; ++i)
        printf("out[%d] = %f\n", i, h_out[i]);

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```