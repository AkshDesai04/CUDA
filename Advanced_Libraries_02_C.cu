```cpp
/* 
Aim of the program (exact wording from the prompt):
"Create multiple generators on different streams to generate random numbers in parallel for different purposes."

Thought process:
- The goal is to demonstrate how CUDA streams and CURAND generators can be combined to produce random numbers concurrently, each generator operating in its own stream so that the generation can happen in parallel.
- We use the CURAND host API because it allows us to create multiple generators and bind each to a distinct stream via curandSetStream().
- For simplicity, we create three streams and three corresponding generators. Each generator will produce a small array of random floating‑point numbers.
- We allocate device memory for each stream's output, generate the numbers, copy them back to the host, and print a few samples to confirm that the operations ran concurrently.
- Basic error checking is included via a macro to wrap CUDA and CURAND calls.
- The program cleans up by destroying the generators, freeing device memory, and destroying the streams.
*/

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

#define CHECK_CURAND(call)                                                     \
    {                                                                         \
        curandStatus_t err = call;                                            \
        if (err != CURAND_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "CURAND error at %s:%d: %d\n", __FILE__, __LINE__,\
                    err);                                                     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

int main(void) {
    const int numStreams = 3;
    const int numsPerStream = 10; // Number of random numbers each generator will produce

    // Create CUDA streams
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Create CURAND generators and bind them to streams
    curandGenerator_t generators[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CURAND(curandCreateGenerator(&generators[i], CURAND_RNG_PSEUDO_XORWOW));
        // Seed with a different value for each generator
        unsigned long long seed = 1234ULL + i * 100ULL;
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(generators[i], seed));
        // Bind generator to its stream
        CHECK_CURAND(curandSetStream(generators[i], streams[i]));
    }

    // Allocate device memory for each stream
    float *devData[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA(cudaMalloc((void **)&devData[i], numsPerStream * sizeof(float)));
    }

    // Generate random numbers on each stream
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CURAND(curandGenerateUniform(generators[i], devData[i], numsPerStream));
    }

    // Copy results back to host
    float *hostResults[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        hostResults[i] = (float *)malloc(numsPerStream * sizeof(float));
        CHECK_CUDA(cudaMemcpyAsync(hostResults[i], devData[i], numsPerStream * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[i]));
    }

    // Synchronize all streams to ensure copy completion
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    // Print sample results
    for (int i = 0; i < numStreams; ++i) {
        printf("Stream %d random numbers:\n", i);
        for (int j = 0; j < numsPerStream; ++j) {
            printf("  %f\n", hostResults[i][j]);
        }
        printf("\n");
    }

    // Cleanup
    for (int i = 0; i < numStreams; ++i) {
        free(hostResults[i]);
        CHECK_CUDA(cudaFree(devData[i]));
        CHECK_CURAND(curandDestroyGenerator(generators[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }

    printf("All random numbers generated and streams cleaned up successfully.\n");
    return 0;
}
```