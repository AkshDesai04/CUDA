```cpp
/*
Create multiple generators on different streams to generate random numbers in parallel for different purposes.

Thinking:
- The goal is to demonstrate how CUDA streams can be used to run random number generation kernels concurrently
  without interfering with each other.  
- We'll use the cuRAND library, specifically the XORWOW generator, as it can be used on the device without
  any external dependencies.
- For each stream we will allocate its own curandState array and its own output arrays.  
- Two different distributions (uniform and normal) will be generated on each stream to show that the
  same generator can serve different purposes.
- We launch separate kernels on each stream; they will run in parallel on the GPU.
- After kernel execution we copy the first few numbers back to the host and print them to verify
  that everything worked.
- Basic error checking is added to ensure proper CUDA API usage.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <time.h>

#define N 1024          // Number of random numbers per stream
#define NUM_STREAMS 3   // Number of streams/generators

// Error checking macro
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Kernel to initialize curandState for each thread
__global__ void init_curand(curandState *state, unsigned long long seed, unsigned long long seq, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Each thread gets same seed but different sequence number
        curand_init(seed, idx, seq, &state[idx]);
    }
}

// Kernel to generate uniform random numbers in [0,1)
__global__ void generate_uniform(curandState *state, float *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = curand_uniform(&state[idx]); // [0,1)
    }
}

// Kernel to generate normal random numbers with mean=0, stddev=1
__global__ void generate_normal(curandState *state, float *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = curand_normal(&state[idx]); // mean=0, stddev=1
    }
}

int main()
{
    // Allocate arrays for device pointers
    curandState *d_state[NUM_STREAMS];
    float *d_uniform[NUM_STREAMS];
    float *d_normal[NUM_STREAMS];

    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Allocate memory for each stream
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaMalloc((void**)&d_state[i], N * sizeof(curandState)));
        CUDA_CHECK(cudaMalloc((void**)&d_uniform[i], N * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_normal[i], N * sizeof(float)));
    }

    // Launch initialization kernels on each stream
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    unsigned long long baseSeed = (unsigned long long)time(NULL);

    for (int i = 0; i < NUM_STREAMS; ++i) {
        unsigned long long seed = baseSeed + i;   // Different seed per stream
        unsigned long long seq  = 0;             // Sequence number can be 0
        init_curand<<<blocks, threadsPerBlock, 0, streams[i]]>(d_state[i], seed, seq, N);
    }

    // Launch generation kernels on each stream
    for (int i = 0; i < NUM_STREAMS; ++i) {
        generate_uniform<<<blocks, threadsPerBlock, 0, streams[i]>>>(d_state[i], d_uniform[i], N);
        generate_normal<<<blocks, threadsPerBlock, 0, streams[i]>>>(d_state[i], d_normal[i], N);
    }

    // Allocate host memory for results
    float *h_uniform[NUM_STREAMS];
    float *h_normal[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        h_uniform[i] = (float*)malloc(N * sizeof(float));
        h_normal[i]  = (float*)malloc(N * sizeof(float));
    }

    // Copy results back to host
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(h_uniform[i], d_uniform[i], N * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(h_normal[i], d_normal[i], N * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[i]));
    }

    // Synchronize streams to ensure all copies are complete
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    // Print first 5 numbers from each stream
    for (int i = 0; i < NUM_STREAMS; ++i) {
        printf("Stream %d: Uniform[0..4] = ", i);
        for (int j = 0; j < 5; ++j) printf("%f ", h_uniform[i][j]);
        printf("\n");

        printf("Stream %d: Normal[0..4]  = ", i);
        for (int j = 0; j < 5; ++j) printf("%f ", h_normal[i][j]);
        printf("\n\n");
    }

    // Clean up
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaFree(d_state[i]));
        CUDA_CHECK(cudaFree(d_uniform[i]));
        CUDA_CHECK(cudaFree(d_normal[i]));
        free(h_uniform[i]);
        free(h_normal[i]);
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    return 0;
}
```