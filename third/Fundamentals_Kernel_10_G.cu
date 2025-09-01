```cpp
/*
Aim of the program:
Compare passing a struct with 3 floats vs. passing 3 separate float arguments to a kernel. Is there a performance difference? (Likely negligible, but good to think about).

Thinking:
When a kernel is launched, its arguments are copied from the host to the device, typically into registers or local memory. A struct of three floats is essentially 12 bytes, same as three separate floats. Passing the struct by value or three floats individually results in the same amount of data being transferred. The CUDA compiler will usually pack the struct into registers just as it would pack separate floats. The difference, if any, may arise from alignment or padding of the struct, but for a simple struct of three consecutive floats there is no padding, so the layout is identical to three separate float arguments.

The kernel itself will load the arguments into registers. Since the operations are trivial, the dominant cost is kernel launch overhead and the time spent in device code, not the argument copy. Therefore we expect negligible performance difference. However, to be thorough we measure the execution time of two kernels: one receiving a struct, the other receiving three floats. The kernels perform a simple computation on an array so that they are not optimized away. We run each kernel many times and average the timing to reduce noise. If a measurable difference exists, it will show up in the timing results.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define checkCuda(call)                                                      \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",             \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Simple struct of three floats
struct MyTriple {
    float x;
    float y;
    float z;
};

// Kernel that receives a struct
__global__ void kernel_struct(float *out, const MyTriple tr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Simple computation to avoid optimization away
        out[idx] = tr.x * idx + tr.y * idx + tr.z * idx;
    }
}

// Kernel that receives three separate floats
__global__ void kernel_three(float *out, float a, float b, float c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a * idx + b * idx + c * idx;
    }
}

int main() {
    const int N = 1 << 20;          // Number of elements
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ITERATIONS = 1000;    // Number of times to run each kernel

    // Host array (not used beyond allocation to ensure proper memory allocation)
    float *h_out = (float*)malloc(N * sizeof(float));
    if (!h_out) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Device array
    float *d_out;
    checkCuda(cudaMalloc((void**)&d_out, N * sizeof(float)));

    // Create events for timing
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    // Struct instance
    MyTriple h_tr = {1.0f, 2.0f, 3.0f};

    // ---- Measure kernel_struct ----
    float time_struct = 0.0f;
    for (int i = 0; i < ITERATIONS; ++i) {
        checkCuda(cudaEventRecord(start, 0));
        kernel_struct<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, h_tr, N);
        checkCuda(cudaEventRecord(stop, 0));
        checkCuda(cudaEventSynchronize(stop));
        float milliseconds = 0;
        checkCuda(cudaEventElapsedTime(&milliseconds, start, stop));
        time_struct += milliseconds;
    }
    time_struct /= ITERATIONS; // average

    // ---- Measure kernel_three ----
    float time_three = 0.0f;
    for (int i = 0; i < ITERATIONS; ++i) {
        checkCuda(cudaEventRecord(start, 0));
        kernel_three<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, 1.0f, 2.0f, 3.0f, N);
        checkCuda(cudaEventRecord(stop, 0));
        checkCuda(cudaEventSynchronize(stop));
        float milliseconds = 0;
        checkCuda(cudaEventElapsedTime(&milliseconds, start, stop));
        time_three += milliseconds;
    }
    time_three /= ITERATIONS; // average

    // Copy back result to host to prevent compiler optimizations
    checkCuda(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Output results
    printf("Average kernel execution time over %d iterations:\n", ITERATIONS);
    printf("kernel_struct: %.6f ms\n", time_struct);
    printf("kernel_three:  %.6f ms\n", time_three);

    // Clean up
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    checkCuda(cudaFree(d_out));
    free(h_out);

    return 0;
}
```