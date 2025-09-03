```cuda
/*
Implement a 1D convolution (or a 1D stencil) using shared memory. Each thread block should load a segment of the input array into shared memory, including 'ghost' cells at the boundaries needed by the threads at the edges of the block. Use __syncthreads() before computation.

Thinking:
- We choose a simple 3‑point stencil (a*x[i-1] + b*x[i] + c*x[i+1]) with coefficients a=0.25, b=0.5, c=0.25.
- For each block, we load the main segment plus two halo elements on each side. 
- The shared memory size is blockDim.x + 2*halo. The halo size is 1 for a 3‑point stencil.
- Ghost elements for the first and last blocks are zero‑padded if the global index falls outside the array bounds.
- Synchronize with __syncthreads() before performing the convolution to ensure all shared memory data is available.
- Host code allocates input data, copies it to device, launches the kernel, copies back results, and prints a few sample values.
- Basic error checking is included to verify CUDA calls.
- The kernel uses a dynamic shared memory declaration (`extern __shared__ float sdata[];`).
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

__global__ void conv1d_shared(const float *input, float *output, int N)
{
    // 3‑point stencil: y[i] = 0.25*x[i-1] + 0.5*x[i] + 0.25*x[i+1]
    const float a = 0.25f, b = 0.5f, c = 0.25f;
    const int halo = 1;  // number of ghost cells on each side

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;

    // Load the main element
    if (globalIdx < N)
        sdata[tid + halo] = input[globalIdx];
    else
        sdata[tid + halo] = 0.0f;   // out of bounds guard

    // Load left ghost cell
    if (tid < halo) {
        int gidx = globalIdx - halo;
        if (gidx >= 0 && gidx < N)
            sdata[tid] = input[gidx];
        else
            sdata[tid] = 0.0f; // zero padding
    }

    // Load right ghost cell
    if (tid >= blockDim.x - halo) {
        int gidx = globalIdx + halo;
        if (gidx >= 0 && gidx < N)
            sdata[tid + 2 * halo] = input[gidx];
        else
            sdata[tid + 2 * halo] = 0.0f; // zero padding
    }

    __syncthreads();

    // Perform convolution only for valid indices
    if (globalIdx < N) {
        float left  = sdata[tid];           // x[i-1]
        float center= sdata[tid + halo];    // x[i]
        float right = sdata[tid + 2 * halo]; // x[i+1]
        output[globalIdx] = a * left + b * center + c * right;
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const int BLOCK_SIZE = 256;
    const int halo = 1;
    const int sharedMemSize = (BLOCK_SIZE + 2 * halo) * sizeof(float);

    size_t sizeBytes = N * sizeof(float);

    float *h_input = (float*)malloc(sizeBytes);
    float *h_output = (float*)malloc(sizeBytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input with random data
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i)
        h_input[i] = (float)(rand()) / RAND_MAX;

    float *d_input = NULL;
    float *d_output = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_input, sizeBytes));
    CHECK_CUDA(cudaMalloc((void**)&d_output, sizeBytes));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeBytes, cudaMemcpyHostToDevice));

    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    conv1d_shared<<<grid, block, sharedMemSize>>>(d_input, d_output, N);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(h_output, d_output, sizeBytes, cudaMemcpyDeviceToHost));

    // Print first 10 results for verification
    printf("Index\tInput\t\tOutput\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d\t%f\t%f\n", i, h_input[i], h_output[i]);
    }

    // Clean up
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return EXIT_SUCCESS;
}
```