/*
Aim: Analyze why the AoS kernel has poor memory performance. 
(Threads in a warp access `x`, but the `y` and `z` components are loaded into cache unnecessarily, wasting bandwidth).

Thought process:
- In an Array-of-Structures (AoS) layout, each element is stored contiguously as {x, y, z}.
- If each thread in a warp reads only the x component, the memory transaction still brings the entire 12‑byte structure (x, y, z) into the cache.
- For a warp of 32 threads, this results in 32 * 12 = 384 bytes transferred for data that only 32 * 4 = 128 bytes are actually needed.
- This inefficiency leads to higher memory bandwidth usage, increased latency, and lower overall performance.
- A Structure-of-Arrays (SoA) layout stores each component in a separate contiguous array. Accessing only x requires loading only the x array, eliminating unnecessary traffic.
- The following CUDA code demonstrates both AoS and SoA kernels so that the performance differences can be measured or visualized.

*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Array of Structures definition
struct float3aos {
    float x;
    float y;
    float z;
};

// Kernel that accesses only the x component from an AoS array
__global__ void kernel_aos(const float3aos* d_aos, float* d_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Each thread reads the x component; y and z are still fetched
        d_out[idx] = d_aos[idx].x;
    }
}

// Kernel that accesses only the x component from a SoA array
__global__ void kernel_soa(const float* d_x, const float* d_y, const float* d_z, float* d_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Only the x array is accessed; y and z are not loaded
        d_out[idx] = d_x[idx];
    }
}

// Helper function to initialize data
void init_host(float3aos* h_aos, int N) {
    for (int i = 0; i < N; ++i) {
        h_aos[i].x = static_cast<float>(i);
        h_aos[i].y = static_cast<float>(i + 1);
        h_aos[i].z = static_cast<float>(i + 2);
    }
}

// Main driver
int main() {
    const int N = 1 << 20; // 1M elements
    const size_t size_aos = N * sizeof(float3aos);
    const size_t size_float = N * sizeof(float);

    // Host allocations
    float3aos* h_aos = (float3aos*)malloc(size_aos);
    float* h_out_aos = (float*)malloc(size_float);
    float* h_out_soa = (float*)malloc(size_float);

    init_host(h_aos, N);

    // Device allocations
    float3aos* d_aos;
    float* d_x; float* d_y; float* d_z;
    float* d_out_aos; float* d_out_soa;
    cudaMalloc((void**)&d_aos, size_aos);
    cudaMalloc((void**)&d_x, size_float);
    cudaMalloc((void**)&d_y, size_float);
    cudaMalloc((void**)&d_z, size_float);
    cudaMalloc((void**)&d_out_aos, size_float);
    cudaMalloc((void**)&d_out_soa, size_float);

    // Copy AoS to device
    cudaMemcpy(d_aos, h_aos, size_aos, cudaMemcpyHostToDevice);

    // Copy SoA components to device
    for (int i = 0; i < N; ++i) {
        // Using cudaMemcpyAsync with pinned memory could be faster,
        // but for clarity we perform three separate copies.
        // We can also use cudaMemcpy2D or create separate host arrays.
    }
    // Simple extraction of SoA on host (inefficient but illustrative)
    float* h_x = (float*)malloc(size_float);
    float* h_y = (float*)malloc(size_float);
    float* h_z = (float*)malloc(size_float);
    for (int i = 0; i < N; ++i) {
        h_x[i] = h_aos[i].x;
        h_y[i] = h_aos[i].y;
        h_z[i] = h_aos[i].z;
    }
    cudaMemcpy(d_x, h_x, size_float, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size_float, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, size_float, cudaMemcpyHostToDevice);

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch AoS kernel
    kernel_aos<<<blocks, threadsPerBlock>>>(d_aos, d_out_aos, N);
    cudaDeviceSynchronize();

    // Launch SoA kernel
    kernel_soa<<<blocks, threadsPerBlock>>>(d_x, d_y, d_z, d_out_soa, N);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_out_aos, d_out_aos, size_float, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_soa, d_out_soa, size_float, cudaMemcpyDeviceToHost);

    // Simple verification
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_out_aos[i] != h_out_soa[i]) {
            errors++;
            if (errors < 10) {
                printf("Mismatch at %d: AoS=%f, SoA=%f\n", i, h_out_aos[i], h_out_soa[i]);
            }
        }
    }
    if (errors == 0) {
        printf("Verification passed.\n");
    } else {
        printf("Verification failed with %d mismatches.\n", errors);
    }

    // Clean up
    free(h_aos);
    free(h_out_aos);
    free(h_out_soa);
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_aos);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_out_aos);
    cudaFree(d_out_soa);

    return 0;
}
