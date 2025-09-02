/*
Aim of the program: Profile and time both versions and report the speedup of SoA over AoS.

The idea is to show how struct‑of‑arrays (SoA) can provide better memory coalescing and hence speed up
memory‑bound kernels compared to array‑of‑structs (AoS).  The program does the following:

1.  Defines a simple data structure for an AoS: a struct containing two floats.
2.  Allocates large arrays (16 million elements) for both AoS and SoA representations.
3.  Initializes the data on the host, copies it to the device.
4.  Implements two identical kernels: one that operates on an AoS array and one that operates
    on a SoA (two separate float arrays).  Each kernel performs a trivial computation
    (increments the two fields by constants) to keep the focus on memory access patterns.
5.  Times each kernel execution using CUDA events, ensuring proper synchronization.
6.  Computes and prints the elapsed times and the speedup factor of the SoA version over the AoS
    version.
7.  Cleans up all allocated resources.

By keeping the computation simple, the main performance difference comes from how the GPU
accesses memory: AoS requires fetching two floats that are interleaved, whereas SoA allows
coalesced access to one field at a time.  The program demonstrates this with measurable
timing outputs.

The code below is a complete .cu file that can be compiled with nvcc.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Number of elements (adjustable)
const size_t N = 1 << 24; // 16,777,216 elements

// AoS data structure
struct AoS {
    float x;
    float y;
};

// Kernel operating on AoS
__global__ void kernel_aos(AoS* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple operation: add constants
        data[idx].x += 1.0f;
        data[idx].y += 2.0f;
    }
}

// Kernel operating on SoA
__global__ void kernel_soa(float* x, float* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] += 1.0f;
        y[idx] += 2.0f;
    }
}

int main(void) {
    // Allocate host memory
    AoS* h_aos = (AoS*)malloc(N * sizeof(AoS));
    float* h_x = (float*)malloc(N * sizeof(float));
    float* h_y = (float*)malloc(N * sizeof(float));

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_aos[i].x = static_cast<float>(i);
        h_aos[i].y = static_cast<float>(i) * 0.5f;
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(i) * 0.5f;
    }

    // Allocate device memory
    AoS* d_aos;
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void**)&d_aos, N * sizeof(AoS)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, N * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_aos, h_aos, N * sizeof(AoS), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Timing variables
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 1. Run AoS kernel
    CHECK_CUDA(cudaEventRecord(start));
    kernel_aos<<<blocksPerGrid, threadsPerBlock>>>(d_aos, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_aos;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_aos, start, stop));

    // 2. Run SoA kernel
    CHECK_CUDA(cudaEventRecord(start));
    kernel_soa<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_soa;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_soa, start, stop));

    // Print results
    printf("AoS kernel time:  %.3f ms\n", elapsed_aos);
    printf("SoA kernel time:  %.3f ms\n", elapsed_soa);
    if (elapsed_soa > 0.0f) {
        printf("Speedup (SoA / AoS): %.2fx\n", elapsed_aos / elapsed_soa);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_aos));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    free(h_aos);
    free(h_x);
    free(h_y);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
