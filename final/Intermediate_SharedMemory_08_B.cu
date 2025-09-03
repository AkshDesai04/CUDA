```cuda
/* 
Aim: Implement a max-finding reduction using the same pattern. 

Thinking:
1. We need a CUDA program that finds the maximum value in an array using parallel reduction.
2. The "same pattern" suggests reusing the typical reduction kernel that operates on shared memory:
   - Each block loads elements into shared memory.
   - The block performs a tree-based reduction, halving the number of active threads each step.
   - Finally, the block writes its partial maximum to a per-block output array.
3. After the first kernel launch, we may have multiple block results. We then launch the same kernel again on this reduced array until only one value remains.
4. The host code handles memory allocation, data transfer, kernel launch configuration, and iteratively reduces until the global maximum is obtained.
5. Error checking will be performed using a simple macro.
6. The program will demonstrate the reduction on a randomly generated array of floats.
7. We keep the code self‑contained in a single .cu file with all necessary includes, definitions, and a `main` function.

This implementation follows the standard parallel reduction pattern used for sum, but replaces addition with the `max` operation. 
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>

// CUDA error checking macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

// Kernel: one block processes 2*blockDim.x elements per iteration
// It writes the block's maximum to the output array
__global__ void reduce_max(const float *input, float *output, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load two elements per thread if within bounds
    float myMax = -FLT_MAX;

    if (idx < n)
        myMax = input[idx];
    if (idx + blockDim.x < n)
        myMax = fmaxf(myMax, input[idx + blockDim.x]);

    sdata[tid] = myMax;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main()
{
    const int N = 1 << 20;  // 1M elements
    const int bytes = N * sizeof(float);

    // Allocate host memory
    float *h_in = (float *)malloc(bytes);
    if (!h_in) { fprintf(stderr, "Host allocation failed\n"); return EXIT_FAILURE; }

    // Initialize data with random floats
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i)
        h_in[i] = (float)rand() / RAND_MAX * 1000.0f; // values between 0 and 1000

    // Allocate device memory
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes)); // temporary buffer

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int currentSize = N;
    float *d_input = d_in;
    float *d_output = d_out;

    // Reduction loop
    while (currentSize > 1)
    {
        int threadsPerBlock = 512;
        int blocks = (currentSize + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
        int sharedMemSize = threadsPerBlock * sizeof(float);

        reduce_max<<<blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_output, currentSize);
        CUDA_CHECK(cudaGetLastError());

        // Prepare for next iteration
        currentSize = blocks;
        // Swap pointers
        float *tmp = d_input;
        d_input = d_output;
        d_output = tmp;
    }

    // Copy result back to host
    float h_max = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_max, d_input, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Maximum value = %f\n", h_max);

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);

    return 0;
}
```