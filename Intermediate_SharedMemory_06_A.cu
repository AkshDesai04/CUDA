```c
/* 
For a 3-point stencil `output[i] = w_l*in[i-1] + w_c*in[i] + w_r*in[i+1]`, each thread needs its neighbors. If a block loads N elements for N threads, it must actually load N+2 elements into shared memory.

Thinking:
1. The stencil requires the left and right neighboring elements for each point. When using shared memory, a block of threads will cooperatively load data from global memory into a shared array, but the first and last elements of the block also need one extra element on each side (the halo). Therefore the shared array must be sized `blockDim.x + 2`.
2. Each thread loads its own data into the middle of the shared array: `smem[threadIdx.x + 1] = in[globalIdx]`.
3. The left halo can be loaded by thread 0 of each block (`smem[0]`) and the right halo by the last thread (`smem[blockDim.x + 1]`). Bounds checks are needed for the array edges.
4. After loading, threads synchronize with `__syncthreads()` to ensure all shared memory data is ready before computing the stencil.
5. The kernel writes the result back to global memory. Boundary elements (first and last of the entire array) are handled specially (here we set them to zero for simplicity).
6. In the host code, we allocate host and device arrays, initialize input, launch the kernel, and copy results back for verification.
7. Error checking macros are used to catch CUDA API errors.
8. The code is written in C/C++ CUDA syntax and can be compiled with `nvcc stencil.cu -o stencil`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                \
    {                                                                   \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    }

__global__ void stencil3pt(const float *in, float *out, int N,
                           float w_l, float w_c, float w_r)
{
    extern __shared__ float smem[];   // size: blockDim.x + 2

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx  = threadIdx.x + 1;   // shift by 1 for halo

    // Load the main element
    if (globalIdx < N)
        smem[localIdx] = in[globalIdx];
    else
        smem[localIdx] = 0.0f;   // out-of-bounds guard

    // Load left halo (only thread 0 does this)
    if (threadIdx.x == 0) {
        int leftIdx = globalIdx - 1;
        smem[0] = (leftIdx >= 0) ? in[leftIdx] : 0.0f;
    }

    // Load right halo (only last thread does this)
    if (threadIdx.x == blockDim.x - 1) {
        int rightIdx = globalIdx + 1;
        smem[blockDim.x + 1] = (rightIdx < N) ? in[rightIdx] : 0.0f;
    }

    __syncthreads();

    // Compute stencil if within bounds (skip first and last global elements)
    if (globalIdx > 0 && globalIdx < N - 1) {
        float left  = smem[localIdx - 1];
        float center = smem[localIdx];
        float right = smem[localIdx + 1];
        out[globalIdx] = w_l * left + w_c * center + w_r * right;
    }
    // For boundaries, we can set to zero or copy input directly
    else if (globalIdx < N) {
        out[globalIdx] = 0.0f; // simple choice for edges
    }
}

int main(void)
{
    const int N = 1024;              // array size
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *h_in  = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));

    // Initialize input with some values, e.g., sine wave or simple pattern
    for (int i = 0; i < N; ++i) {
        h_in[i] = sinf(i * 0.01f);
    }

    // Device pointers
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc((void**)&d_in,  N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_out, N * sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Define weights
    float w_l = 0.25f;
    float w_c = 0.5f;
    float w_r = 0.25f;

    // Launch kernel
    size_t sharedMemSize = (threadsPerBlock + 2) * sizeof(float);
    stencil3pt<<<blocks, threadsPerBlock, sharedMemSize>>>(d_in, d_out, N,
                                                           w_l, w_c, w_r);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print some results
    printf("Index\tInput\t\tOutput\n");
    for (int i = 0; i < N; i += N/10) {
        printf("%5d\t%8.4f\t%8.4f\n", i, h_in[i], h_out[i]);
    }

    // Clean up
    free(h_in);
    free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
```