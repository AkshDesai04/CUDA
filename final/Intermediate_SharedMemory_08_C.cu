/*
Implement a min-finding reduction.
I began by outlining the goal: create a CUDA program in C that computes the minimum element of an integer array using a reduction kernel. 
The typical approach is a two‑stage reduction: first, each block reduces its portion of the array into a block‑level minimum stored in shared memory, then writes that minimum to a global array. 
After all blocks finish, the host can either launch another kernel to reduce the block minima, or simply perform a final reduction on the CPU. 
I opted for a CPU final reduction for simplicity and to keep the CUDA code focused on the intra‑block reduction. 
The kernel uses shared memory and iterative halving of active threads, employing a standard reduction pattern. 
To make the program self‑contained, I included memory allocation, error checking, a simple test array, and output of the resulting minimum. 
The code uses integer data types and a fixed block size of 256 threads, but it can be adapted easily for other sizes or data types. 
The program is fully compilable with nvcc and demonstrates a straightforward min‑finding reduction in CUDA C. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCK_SIZE 256

// CUDA error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel: each block finds the minimum of its chunk and writes it to blockMins
__global__ void reduceMinKernel(const int *d_in, int *d_blockMins, int N) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory, handling out-of-bounds
    sdata[tid] = (idx < N) ? d_in[idx] : INT_MAX;
    __syncthreads();

    // Intra‑block reduction: find minimum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (idx + s) < N) {
            if (sdata[tid + s] < sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block's minimum to global memory
    if (tid == 0) {
        d_blockMins[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char **argv) {
    // Size of the array
    const int N = 1 << 20;  // 1M elements
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_in = (int *)malloc(size);
    if (!h_in) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize array with random integers
    for (int i = 0; i < N; ++i) {
        h_in[i] = rand() % 10000;  // random numbers between 0 and 9999
    }

    // Allocate device memory
    int *d_in;
    int *d_blockMins;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMalloc((void **)&d_in, size));
    CUDA_CHECK(cudaMalloc((void **)&d_blockMins, numBlocks * sizeof(int)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Launch kernel
    size_t sharedMemSize = BLOCK_SIZE * sizeof(int);
    reduceMinKernel<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_in, d_blockMins, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy block minima back to host
    int *h_blockMins = (int *)malloc(numBlocks * sizeof(int));
    if (!h_blockMins) {
        fprintf(stderr, "Failed to allocate host memory for block minima.\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(h_blockMins, d_blockMins, numBlocks * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Final reduction on CPU
    int minVal = INT_MAX;
    for (int i = 0; i < numBlocks; ++i) {
        if (h_blockMins[i] < minVal)
            minVal = h_blockMins[i];
    }

    printf("Minimum value found: %d\n", minVal);

    // Clean up
    free(h_in);
    free(h_blockMins);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_blockMins));

    return EXIT_SUCCESS;
}
