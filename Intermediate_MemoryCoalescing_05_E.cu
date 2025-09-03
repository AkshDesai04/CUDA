```cpp
/*
Aim of the program:
Profile and time both versions and report the speedup of SoA over AoS.

Thinking:
The goal is to compare two data layouts – Array of Structures (AoS) and Structure of Arrays (SoA) –
in a CUDA kernel that performs a simple arithmetic operation on 3-component vectors.
We will measure the execution time of each kernel using CUDA events and compute the speedup
as AoS time divided by SoA time.

Design choices:
1. Use a simple operation that accesses all three components: compute the Euclidean norm
   sqrt(x^2 + y^2 + z^2). This ensures that the memory access pattern is important.
2. Allocate a large dataset (e.g., 16 million elements) to make the difference visible.
3. Implement two kernels: kernelAoS and kernelSoA, each processing the same data.
4. Use cudaEvent_t to time only the kernel launch (excluding memory copies).
5. Verify correctness by comparing a few sample results from both layouts.
6. Output the measured times and the computed speedup.

The program is self-contained, compilable with nvcc, and follows CUDA best practices.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Number of elements (3-component vectors)
const size_t N = 1 << 24; // 16,777,216

// Structure for AoS layout
struct float3_AoS {
    float x;
    float y;
    float z;
};

// Kernel for AoS
__global__ void kernelAoS(const float3_AoS* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float3_AoS &v = in[idx];
        out[idx] = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }
}

// Kernel for SoA
__global__ void kernelSoA(const float* __restrict__ x, const float* __restrict__ y,
                          const float* __restrict__ z, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrtf(x[idx] * x[idx] + y[idx] * y[idx] + z[idx] * z[idx]);
    }
}

int main() {
    printf("=== AoS vs SoA Performance Comparison ===\n");

    // Host allocations
    float3_AoS* h_aos = (float3_AoS*)malloc(N * sizeof(float3_AoS));
    float* h_x = (float*)malloc(N * sizeof(float));
    float* h_y = (float*)malloc(N * sizeof(float));
    float* h_z = (float*)malloc(N * sizeof(float));
    float* h_out_aos = (float*)malloc(N * sizeof(float));
    float* h_out_soa = (float*)malloc(N * sizeof(float));

    // Initialize data
    for (size_t i = 0; i < N; ++i) {
        float x = static_cast<float>(rand()) / RAND_MAX;
        float y = static_cast<float>(rand()) / RAND_MAX;
        float z = static_cast<float>(rand()) / RAND_MAX;
        h_aos[i] = {x, y, z};
        h_x[i] = x;
        h_y[i] = y;
        h_z[i] = z;
    }

    // Device allocations
    float3_AoS* d_aos;
    float* d_x;
    float* d_y;
    float* d_z;
    float* d_out;

    CUDA_CHECK(cudaMalloc((void**)&d_aos, N * sizeof(float3_AoS)));
    CUDA_CHECK(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_z, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_aos, h_aos, N * sizeof(float3_AoS), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Time AoS kernel
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernelAoS<<<blocks, threadsPerBlock>>>(d_aos, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_aos = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_aos, start, stop));

    // Copy AoS result back
    CUDA_CHECK(cudaMemcpy(h_out_aos, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Time SoA kernel
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernelSoA<<<blocks, threadsPerBlock>>>(d_x, d_y, d_z, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_soa = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_soa, start, stop));

    // Copy SoA result back
    CUDA_CHECK(cudaMemcpy(h_out_soa, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few results
    bool ok = true;
    for (size_t i = 0; i < 10; ++i) {
        if (fabs(h_out_aos[i] - h_out_soa[i]) > 1e-5f) {
            printf("Mismatch at index %zu: AoS=%f, SoA=%f\n", i, h_out_aos[i], h_out_soa[i]);
            ok = false;
            break;
        }
    }
    if (ok) printf("Verification: OK\n");
    else printf("Verification: FAILED\n");

    // Report times and speedup
    printf("AoS kernel time: %.3f ms\n", time_aos);
    printf("SoA kernel time: %.3f ms\n", time_soa);
    if (time_soa > 0.0f) {
        printf("Speedup (AoS / SoA): %.2fx\n", time_aos / time_soa);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_aos));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_aos);
    free(h_x);
    free(h_y);
    free(h_z);
    free(h_out_aos);
    free(h_out_soa);

    return 0;
}
```