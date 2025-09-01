```cuda
/* 
Aim of the program (as passed in the prompt word for word):

Write the computation phase: after synchronizing, each thread computes its dot product using data exclusively from the shared memory tiles.

Thinking:
To implement this computation phase in CUDA we create a kernel that loads a tile of two input vectors into shared memory. 
After a __syncthreads() barrier, each thread will iterate over the entire tile using only the shared memory copies to compute its 
own partial dot product.  The partial results are written to a global array.  The host then reduces these partial sums to obtain
the final dot product.  This approach demonstrates how each thread can work exclusively with shared memory data after synchronization.

The code below:
- defines a kernel `dotProductKernel` that performs the described phase.
- loads elements into shared memory, synchronizes, computes partial dot products,
  and stores the result per thread.
- contains a simple host routine that allocates memory, launches the kernel,
  and performs a final reduction on the CPU.
- uses a fixed block size (256 threads) for illustration.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256   // number of threads per block
#define TILE_SIZE  BLOCK_SIZE

// Kernel that performs the computation phase described above
__global__ void dotProductKernel(const float *a, const float *b, float *partial, int N)
{
    __shared__ float s_a[TILE_SIZE];
    __shared__ float s_b[TILE_SIZE];

    int idx = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Load data into shared memory, handling bounds
    if (idx < N) {
        s_a[threadIdx.x] = a[idx];
        s_b[threadIdx.x] = b[idx];
    } else {
        s_a[threadIdx.x] = 0.0f;
        s_b[threadIdx.x] = 0.0f;
    }

    // Synchronize to ensure all shared memory loads are complete
    __syncthreads();

    // Each thread computes a dot product over the entire tile using only shared memory
    float sum = 0.0f;
    for (int i = 0; i < TILE_SIZE; ++i) {
        sum += s_a[i] * s_b[i];
    }

    // Optional second sync if needed before writing out (not strictly necessary here)
    __syncthreads();

    // Store the partial result per thread
    partial[blockIdx.x * BLOCK_SIZE + threadIdx.x] = sum;
}

// Utility function to check CUDA errors
inline void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    const int N = 1 << 20; // Example size: 1M elements
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);

    // Initialize data
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_partial;
    checkCuda(cudaMalloc((void **)&d_a, bytes), "malloc d_a");
    checkCuda(cudaMalloc((void **)&d_b, bytes), "malloc d_b");
    checkCuda(cudaMalloc((void **)&d_partial, bytes), "malloc d_partial");

    // Copy data to device
    checkCuda(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "copy h_a to d_a");
    checkCuda(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "copy h_b to d_b");

    // Determine grid dimensions
    int gridSize = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 block(BLOCK_SIZE);
    dim3 grid(gridSize);

    // Launch kernel
    dotProductKernel<<<grid, block>>>(d_a, d_b, d_partial, N);
    checkCuda(cudaGetLastError(), "kernel launch");

    // Copy partial results back to host
    float *h_partial = (float *)malloc(bytes);
    checkCuda(cudaMemcpy(h_partial, d_partial, bytes, cudaMemcpyDeviceToHost), "copy d_partial to h_partial");

    // Final reduction on host
    double dot = 0.0;
    for (int i = 0; i < N; ++i) {
        dot += h_partial[i];
    }

    printf("Dot product = %f\n", dot);

    // Clean up
    free(h_a);
    free(h_b);
    free(h_partial);
    checkCuda(cudaFree(d_a), "free d_a");
    checkCuda(cudaFree(d_b), "free d_b");
    checkCuda(cudaFree(d_partial), "free d_partial");

    return 0;
}
```