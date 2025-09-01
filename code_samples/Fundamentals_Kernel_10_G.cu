```cpp
/*
 * Aim of the program:
 * Compare passing a struct with 3 floats vs. passing 3 separate float arguments to a CUDA kernel.
 * Is there a performance difference? (Likely negligible, but good to think about).
 *
 * Thinking:
 * 1. Define a simple struct containing three floats.
 * 2. Write two kernels:
 *    - kernelStruct takes the struct as a parameter.
 *    - kernelFloats takes three float parameters.
 * 3. In each kernel, perform a small arithmetic workload that repeats many times to make timing measurable.
 * 4. On the host, launch both kernels with identical grid/block configuration and time each using CUDA events.
 * 5. Copy a single result value back to host and print timings.
 * 6. Use cudaMallocManaged or simple device memory for results.
 * 7. Because the struct is passed by value and is only 12 bytes, the compiler will generate code equivalent to passing three separate floats.
 * 8. The difference in execution time should be negligible; any observed difference would likely be within noise.
 * 9. Compile with -arch=sm_XX (e.g., sm_60) to ensure modern GPU features.
 * 10. Include error checking macro to simplify code.
 */

#include <cstdio>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(err)                                        \
    do {                                                       \
        cudaError_t err__ = (err);                             \
        if (err__ != cudaSuccess) {                            \
            fprintf(stderr, "CUDA error: %s (err %d) at %s:%d\n", \
                    cudaGetErrorString(err__), err__, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

// Define the struct with 3 floats
struct MyStruct {
    float a;
    float b;
    float c;
};

// Kernel that receives the struct
__global__ void kernelStruct(const MyStruct s, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    // Perform a dummy workload to have measurable execution time
    for (int i = 0; i < 1000; ++i) {
        sum += s.a + s.b + s.c;
    }
    out[idx] = sum;
}

// Kernel that receives 3 separate floats
__global__ void kernelFloats(float a, float b, float c, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < 1000; ++i) {
        sum += a + b + c;
    }
    out[idx] = sum;
}

int main() {
    const int numThreads = 1 << 20; // 1M threads
    const int numBlocks = 1;        // single block for simplicity
    const int N = numThreads * numBlocks;

    // Allocate output arrays
    float *d_outStruct, *d_outFloats;
    CUDA_CHECK(cudaMalloc((void**)&d_outStruct, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_outFloats, N * sizeof(float)));

    // Prepare struct data
    MyStruct h_s{1.0f, 2.0f, 3.0f};
    MyStruct d_s;
    CUDA_CHECK(cudaMemcpy(&d_s, &h_s, sizeof(MyStruct), cudaMemcpyHostToDevice));

    // Events for timing
    cudaEvent_t startStruct, stopStruct;
    cudaEvent_t startFloats, stopFloats;
    CUDA_CHECK(cudaEventCreate(&startStruct));
    CUDA_CHECK(cudaEventCreate(&stopStruct));
    CUDA_CHECK(cudaEventCreate(&startFloats));
    CUDA_CHECK(cudaEventCreate(&stopFloats));

    // Launch kernelStruct
    CUDA_CHECK(cudaEventRecord(startStruct));
    kernelStruct<<<numBlocks, numThreads>>>(d_s, d_outStruct);
    CUDA_CHECK(cudaEventRecord(stopStruct));

    // Launch kernelFloats
    CUDA_CHECK(cudaEventRecord(startFloats));
    kernelFloats<<<numBlocks, numThreads>>>(h_s.a, h_s.b, h_s.c, d_outFloats);
    CUDA_CHECK(cudaEventRecord(stopFloats));

    // Wait for kernels to finish
    CUDA_CHECK(cudaEventSynchronize(stopStruct));
    CUDA_CHECK(cudaEventSynchronize(stopFloats));

    // Compute elapsed time
    float msStruct = 0.0f, msFloats = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msStruct, startStruct, stopStruct));
    CUDA_CHECK(cudaEventElapsedTime(&msFloats, startFloats, stopFloats));

    printf("Kernel with struct time:   %.3f ms\n", msStruct);
    printf("Kernel with 3 floats time: %.3f ms\n", msFloats);

    // Optional: copy back one value to verify kernels executed
    float h_out;
    CUDA_CHECK(cudaMemcpy(&h_out, d_outStruct, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Sample output from struct kernel: %.3f\n", h_out);
    CUDA_CHECK(cudaMemcpy(&h_out, d_outFloats, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Sample output from floats kernel: %.3f\n", h_out);

    // Cleanup
    CUDA_CHECK(cudaFree(d_outStruct));
    CUDA_CHECK(cudaFree(d_outFloats));
    CUDA_CHECK(cudaEventDestroy(startStruct));
    CUDA_CHECK(cudaEventDestroy(stopStruct));
    CUDA_CHECK(cudaEventDestroy(startFloats));
    CUDA_CHECK(cudaEventDestroy(stopFloats));

    return 0;
}
```