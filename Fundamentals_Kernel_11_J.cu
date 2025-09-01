```cuda
/*
Aim of the program:
Define a struct with many members. Time the difference between copying the whole struct vs. using cudaMemcpy on individual members (the former is much better).

Thinking:
- Create a large struct with several member arrays of different types to mimic a realistic data structure.
- Allocate this struct on both host and device.
- Initialize the host struct with some data.
- Measure the time to copy the entire struct in one cudaMemcpy call.
- Measure the time to copy each member array separately using individual cudaMemcpy calls.
- Use CUDA events for precise timing.
- Compare the elapsed times and print the results.
- Ensure proper error checking and cleanup.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Define a large struct with many members
struct BigStruct {
    float  f[64];   // 64 floats
    double d[32];   // 32 doubles
    int    i[128];  // 128 ints
};

int main(void) {
    // Seed random number generator
    srand((unsigned)time(NULL));

    // Allocate and initialize host struct
    BigStruct h_struct;
    for (int idx = 0; idx < 64; ++idx) h_struct.f[idx] = (float)rand() / RAND_MAX;
    for (int idx = 0; idx < 32; ++idx) h_struct.d[idx] = (double)rand() / RAND_MAX;
    for (int idx = 0; idx < 128; ++idx) h_struct.i[idx] = rand();

    // Allocate device struct
    BigStruct* d_struct;
    CUDA_CHECK(cudaMalloc((void**)&d_struct, sizeof(BigStruct)));

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float wholeTimeMs = 0.0f;
    float memberTimeMs = 0.0f;

    // --- Copy whole struct in one go ---
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(d_struct, &h_struct, sizeof(BigStruct),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&wholeTimeMs, start, stop));

    // --- Copy each member separately ---
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Copy array f
    CUDA_CHECK(cudaMemcpy(&d_struct->f, &h_struct.f, sizeof(h_struct.f),
                          cudaMemcpyHostToDevice));
    // Copy array d
    CUDA_CHECK(cudaMemcpy(&d_struct->d, &h_struct.d, sizeof(h_struct.d),
                          cudaMemcpyHostToDevice));
    // Copy array i
    CUDA_CHECK(cudaMemcpy(&d_struct->i, &h_struct.i, sizeof(h_struct.i),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&memberTimeMs, start, stop));

    // Print results
    printf("Time to copy whole struct (one cudaMemcpy): %.3f ms\n", wholeTimeMs);
    printf("Time to copy members separately (three cudaMemcpy): %.3f ms\n",
           memberTimeMs);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_struct));

    return 0;
}
```