```cpp
/*
Aim: Write a CUDA program that accesses a struct array struct {float x,y,z;} and compares performance of an Array of Structs (AoS) access pattern (my_structs[i].x) versus a Struct of Arrays (SoA) pattern (my_xs[i], my_ys[i], my_zs[i]). SoA is usually much better for coalescing.

Thinking:
- Define a simple Vec3 struct for AoS.
- For SoA we keep three separate float arrays.
- We will launch two kernels: one that reads from AoS and writes a sum of the components to a destination array, and another that does the same with SoA arrays.
- To measure performance, use cudaEvent to time kernel execution only.
- Use a large problem size (e.g., 1<<20 elements) to get a measurable difference.
- Use block size of 256 threads; grid size computed accordingly.
- After kernels, we copy a few results back to host to ensure the GPU actually performed the computation (preventing lazy execution or kernel removal).
- Print elapsed time for each kernel.
- Include error checking macro for CUDA calls.
- The program is self‑contained in a single .cu file and can be compiled with nvcc.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Structure for Array of Structs
struct Vec3 {
    float x, y, z;
};

// Kernel that operates on AoS
__global__ void kernel_aos(const Vec3* __restrict__ src, float* __restrict__ dst, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = src[idx].x + src[idx].y + src[idx].z;
        dst[idx] = sum;
    }
}

// Kernel that operates on SoA
__global__ void kernel_soa(const float* __restrict__ xs,
                           const float* __restrict__ ys,
                           const float* __restrict__ zs,
                           float* __restrict__ dst,
                           int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = xs[idx] + ys[idx] + zs[idx];
        dst[idx] = sum;
    }
}

// Helper to fill AoS array with some values
void init_aos(Vec3* h_vecs, int N)
{
    for (int i = 0; i < N; ++i) {
        h_vecs[i].x = (float)i * 0.001f;
        h_vecs[i].y = (float)(i + 1) * 0.002f;
        h_vecs[i].z = (float)(i + 2) * 0.003f;
    }
}

// Helper to fill SoA arrays with same values as AoS
void init_soa(float* h_xs, float* h_ys, float* h_zs, int N)
{
    for (int i = 0; i < N; ++i) {
        h_xs[i] = (float)i * 0.001f;
        h_ys[i] = (float)(i + 1) * 0.002f;
        h_zs[i] = (float)(i + 2) * 0.003f;
    }
}

int main()
{
    const int N = 1 << 20; // 1,048,576 elements
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Host allocations
    Vec3* h_vecs = (Vec3*)malloc(N * sizeof(Vec3));
    float* h_xs = (float*)malloc(N * sizeof(float));
    float* h_ys = (float*)malloc(N * sizeof(float));
    float* h_zs = (float*)malloc(N * sizeof(float));
    float* h_dst = (float*)malloc(N * sizeof(float));

    init_aos(h_vecs, N);
    init_soa(h_xs, h_ys, h_zs, N);

    // Device allocations
    Vec3* d_vecs = nullptr;
    float* d_xs = nullptr, * d_ys = nullptr, * d_zs = nullptr;
    float* d_dst = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_vecs, N * sizeof(Vec3)));
    CHECK_CUDA(cudaMalloc((void**)&d_xs,   N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_ys,   N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_zs,   N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_dst,  N * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_vecs, h_vecs, N * sizeof(Vec3), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_xs, h_xs, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ys, h_ys, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_zs, h_zs, N * sizeof(float), cudaMemcpyHostToDevice));

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ---- AoS kernel timing ----
    CHECK_CUDA(cudaEventRecord(start));
    kernel_aos<<<gridSize, blockSize>>>(d_vecs, d_dst, N);
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_aos = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_aos, start, stop));

    // ---- SoA kernel timing ----
    CHECK_CUDA(cudaEventRecord(start));
    kernel_soa<<<gridSize, blockSize>>>(d_xs, d_ys, d_zs, d_dst, N);
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_soa = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_soa, start, stop));

    // Copy a few results back to verify computation
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("First 5 results from last kernel run (should be same for AoS and SoA):\n");
    for (int i = 0; i < 5; ++i) {
        printf("%f ", h_dst[i]);
    }
    printf("\n");

    // Print timings
    printf("AoS kernel time: %.3f ms\n", ms_aos);
    printf("SoA kernel time: %.3f ms\n", ms_soa);

    // Clean up
    free(h_vecs);
    free(h_xs);
    free(h_ys);
    free(h_zs);
    free(h_dst);

    CHECK_CUDA(cudaFree(d_vecs));
    CHECK_CUDA(cudaFree(d_xs));
    CHECK_CUDA(cudaFree(d_ys));
    CHECK_CUDA(cudaFree(d_zs));
    CHECK_CUDA(cudaFree(d_dst));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```