```cpp
/*
What if the kernel needed to access `x`, `y`, and `z`? `output[i] = d_structs[i].x + d_structs[i].y + d_structs[i].z;`. Compare this to the SoA version. The performance gap might narrow.

Thinking:
- The purpose of this program is to illustrate the performance difference between
  Array-of-Structures (AoS) and Structure-of-Arrays (SoA) layouts when a kernel
  accesses multiple members of a struct. In a realistic scenario, a kernel might
  need to read all fields (`x`, `y`, `z`). This example will compute the sum
  of those three components for each element and write the result to an output
  array.
- The program will:
  1. Allocate host memory for AoS (an array of structs) and SoA (three separate
     arrays for `x`, `y`, `z`).
  2. Initialize the data with random values.
  3. Copy data to the device.
  4. Launch two kernels: one that operates on AoS and one that operates on SoA.
  5. Measure execution time using CUDA events.
  6. Copy results back to host and verify correctness.
  7. Print timing and a basic correctness check.
- The kernels will be simple: each thread processes one element, summing the
  three components and storing the result.
- To keep the example self‑contained, error checking macros and a fixed
  problem size (e.g., 1<<20 elements) are used.
- This program demonstrates that when a kernel accesses all members of a struct,
  the performance difference between AoS and SoA may indeed become smaller,
  because the memory access patterns are more similar.
*/
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Error checking macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Number of elements
const size_t N = 1 << 20; // ~1M elements

// Structure definition for AoS
struct Vec3 {
    float x;
    float y;
    float z;
};

// Kernel operating on AoS
__global__ void sum_aos(const Vec3* d_in, float* d_out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const Vec3& v = d_in[idx];
        d_out[idx] = v.x + v.y + v.z;
    }
}

// Kernel operating on SoA
__global__ void sum_soa(const float* d_x, const float* d_y, const float* d_z, float* d_out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_x[idx] + d_y[idx] + d_z[idx];
    }
}

int main() {
    // Allocate host memory
    Vec3* h_aos = new Vec3[N];
    float* h_x = new float[N];
    float* h_y = new float[N];
    float* h_z = new float[N];
    float* h_out_aos = new float[N];
    float* h_out_soa = new float[N];

    // Initialize data with random values
    srand(42);
    for (size_t i = 0; i < N; ++i) {
        float rx = static_cast<float>(rand()) / RAND_MAX;
        float ry = static_cast<float>(rand()) / RAND_MAX;
        float rz = static_cast<float>(rand()) / RAND_MAX;
        h_aos[i] = {rx, ry, rz};
        h_x[i] = rx;
        h_y[i] = ry;
        h_z[i] = rz;
    }

    // Device pointers
    Vec3* d_aos;
    float *d_x, *d_y, *d_z, *d_out_aos, *d_out_soa;
    CUDA_CHECK(cudaMalloc(&d_aos, N * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_aos, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_soa, N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_aos, h_aos, N * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Timing variables
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ----- AoS kernel -----
    CUDA_CHECK(cudaEventRecord(start, 0));
    sum_aos<<<blocksPerGrid, threadsPerBlock>>>(d_aos, d_out_aos, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_aos;
    CUDA_CHECK(cudaEventElapsedTime(&time_aos, start, stop));

    // ----- SoA kernel -----
    CUDA_CHECK(cudaEventRecord(start, 0));
    sum_soa<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, d_out_soa, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_soa;
    CUDA_CHECK(cudaEventElapsedTime(&time_soa, start, stop));

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_out_aos, d_out_aos, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_soa, d_out_soa, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify correctness
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        if (fabs(h_out_aos[i] - h_out_soa[i]) > 1e-5f) {
            ok = false;
            std::cerr << "Mismatch at index " << i << ": "
                      << h_out_aos[i] << " vs " << h_out_soa[i] << std::endl;
            break;
        }
    }

    std::cout << "Verification: " << (ok ? "PASSED" : "FAILED") << std::endl;
    std::cout << "AoS kernel time: " << time_aos << " ms" << std::endl;
    std::cout << "SoA kernel time: " << time_soa << " ms" << std::endl;

    // Cleanup
    delete[] h_aos;
    delete[] h_x;
    delete[] h_y;
    delete[] h_z;
    delete[] h_out_aos;
    delete[] h_out_soa;

    CUDA_CHECK(cudaFree(d_aos));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_out_aos));
    CUDA_CHECK(cudaFree(d_out_soa));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```