/*
After loading and synchronizing, each thread i computes its result using shared_mem[local_i-1], shared_mem[local_i], and shared_mem[local_i+1].

This program demonstrates a simple 1‑D 3‑point stencil using CUDA shared memory. The kernel loads a block of input data into a shared array, including two halo cells on either side. After a __syncthreads() barrier, each thread computes the average of its element and its two immediate neighbors and writes the result to the output array. Boundary elements (first and last in the global array) are handled separately to avoid out‑of‑bounds accesses. The host code allocates memory, initializes input, launches the kernel, copies back the result, and performs a simple verification by printing a few values. The code also checks for CUDA errors after each API call.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// CUDA kernel: 1D 3‑point stencil
__global__ void stencilKernel(const float* __restrict__ d_in,
                              float* __restrict__ d_out,
                              int N)
{
    // Shared memory with two halo cells
    extern __shared__ float s_data[];

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx  = threadIdx.x + 1;          // offset by 1 for halo

    // Load central data into shared memory
    if (globalIdx < N)
        s_data[localIdx] = d_in[globalIdx];
    else
        s_data[localIdx] = 0.0f; // padding for out‑of‑bounds

    // Load left halo
    if (threadIdx.x == 0) {
        int leftIdx = globalIdx - 1;
        s_data[0] = (leftIdx >= 0) ? d_in[leftIdx] : d_in[globalIdx];
    }

    // Load right halo
    if (threadIdx.x == blockDim.x - 1) {
        int rightIdx = globalIdx + 1;
        s_data[blockDim.x + 1] = (rightIdx < N) ? d_in[rightIdx] : d_in[globalIdx];
    }

    __syncthreads();

    // Compute stencil for interior points only
    if (globalIdx < N) {
        if (globalIdx == 0 || globalIdx == N - 1) {
            // Boundary handling: copy input directly
            d_out[globalIdx] = d_in[globalIdx];
        } else {
            float left  = s_data[localIdx - 1];
            float center = s_data[localIdx];
            float right = s_data[localIdx + 1];
            d_out[globalIdx] = (left + center + right) / 3.0f;
        }
    }
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main(void)
{
    const int N = 1 << 20;  // Example size: 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_in  = (float*)malloc(size);
    float* h_out = (float*)malloc(size);

    // Initialize input with some values
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i) / N;
    }

    // Allocate device memory
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, size));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Determine grid dimensions
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    size_t sharedMemSize = (threadsPerBlock + 2) * sizeof(float);
    stencilKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 values
    printf("First 10 input values:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%0.6f ", h_in[i]);
    }
    printf("\n");

    printf("First 10 output values:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%0.6f ", h_out[i]);
    }
    printf("\n");

    // Clean up
    free(h_in);
    free(h_out);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
