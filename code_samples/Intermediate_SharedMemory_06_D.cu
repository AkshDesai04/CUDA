```cu
/* 
Aim: After loading and synchronizing, each thread i computes its result using shared_mem[local_i-1], shared_mem[local_i], and shared_mem[local_i+1].

Thinking:
- The program demonstrates the use of shared memory for stencil computations.
- It allocates an array of floats, copies to device, launches a kernel that performs a simple 3‑point stencil (average of a pixel and its immediate neighbors).
- Each block loads its segment of the array into shared memory, adding halo elements at both ends.
- Synchronization ensures all data are loaded before any thread accesses the shared memory.
- Each thread then reads shared_mem[local_i-1], shared_mem[local_i], shared_mem[local_i+1] to compute the output.
- Results are copied back and printed.
- Basic CUDA error checking is included.
- The code is written in C‑style CUDA (.cu) and can be compiled with nvcc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define N (1 << 20)          // Number of elements
#define THREADS_PER_BLOCK 256

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel: 3-point stencil using shared memory
__global__ void stencilKernel(float *out, const float *in, int n)
{
    extern __shared__ float shared_mem[];

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int sharedIdx = threadIdx.x + 1;  // offset by 1 for halo on left

    // Guard against out-of-bounds global indices
    if (globalIdx >= n) return;

    // Load central element into shared memory
    shared_mem[sharedIdx] = in[globalIdx];

    // Load left halo element (only thread 0 does this)
    if (threadIdx.x == 0) {
        if (globalIdx > 0)
            shared_mem[0] = in[globalIdx - 1];
        else
            shared_mem[0] = 0.0f;   // boundary condition
    }

    // Load right halo element (only last thread in block does this)
    if (threadIdx.x == blockDim.x - 1) {
        if (globalIdx + 1 < n)
            shared_mem[blockDim.x + 1] = in[globalIdx + 1];
        else
            shared_mem[blockDim.x + 1] = 0.0f;   // boundary condition
    }

    // Ensure all shared memory loads are complete
    __syncthreads();

    // Compute the 3-point average
    float left   = shared_mem[sharedIdx - 1];
    float center = shared_mem[sharedIdx];
    float right  = shared_mem[sharedIdx + 1];

    out[globalIdx] = (left + center + right) / 3.0f;
}

int main(void)
{
    // Host memory allocation
    float *h_in  = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)malloc(N * sizeof(float));

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        h_in[i] = sinf(i * 0.01f);
    }

    // Device memory allocation
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_out, N * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Determine grid and block dimensions
    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Launch kernel
    size_t sharedMemSize = (THREADS_PER_BLOCK + 2) * sizeof(float);
    stencilKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_out, d_in, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print first 10 results for verification
    printf("First 10 results of 3-point stencil:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_out[%d] = %f\n", i, h_out[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return 0;
}
```