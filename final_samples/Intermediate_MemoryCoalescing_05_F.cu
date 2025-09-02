```cpp
/*
Aim of the program: What if the kernel needed to access `x`, `y`, and `z`? `output[i] = d_structs[i].x + d_structs[i].y + d_structs[i].z;`. Compare this to the SoA version. The performance gap might narrow.

Thinking:
- We want to demonstrate the difference between AoS (Array of Structs) and SoA (Struct of Arrays) memory layouts.
- For AoS, we define a struct containing three floats (x, y, z) and allocate an array of these structs on the device.
- For SoA, we allocate three separate device arrays (one for each field) and perform the same sum operation.
- We'll measure the execution time of each kernel using CUDA events to see the performance difference.
- The kernels will be identical in logic: they sum the three fields into an output array.
- We also allocate an output array on the device for each case and copy it back to the host for validation.
- To keep the example simple, we use a large N (e.g., 10 million elements) and a single kernel launch.
- The code includes basic error checking for CUDA calls.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 10000000  // 10 million elements

// Check CUDA errors
inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// AoS structure
struct Vec3 {
    float x;
    float y;
    float z;
};

// Kernel for AoS
__global__ void kernel_aos(const Vec3* __restrict__ d_structs, float* __restrict__ d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_structs[idx].x + d_structs[idx].y + d_structs[idx].z;
    }
}

// Kernel for SoA
__global__ void kernel_soa(const float* __restrict__ d_x,
                           const float* __restrict__ d_y,
                           const float* __restrict__ d_z,
                           float* __restrict__ d_out,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_x[idx] + d_y[idx] + d_z[idx];
    }
}

int main() {
    // Host allocations
    float* h_out_aos = (float*)malloc(N * sizeof(float));
    float* h_out_soa = (float*)malloc(N * sizeof(float));
    if (!h_out_aos || !h_out_soa) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Device allocations for AoS
    Vec3* d_structs = nullptr;
    float* d_out_aos = nullptr;
    checkCuda(cudaMalloc((void**)&d_structs, N * sizeof(Vec3)), "cudaMalloc d_structs");
    checkCuda(cudaMalloc((void**)&d_out_aos, N * sizeof(float)), "cudaMalloc d_out_aos");

    // Device allocations for SoA
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_z = nullptr;
    float* d_out_soa = nullptr;
    checkCuda(cudaMalloc((void**)&d_x, N * sizeof(float)), "cudaMalloc d_x");
    checkCuda(cudaMalloc((void**)&d_y, N * sizeof(float)), "cudaMalloc d_y");
    checkCuda(cudaMalloc((void**)&d_z, N * sizeof(float)), "cudaMalloc d_z");
    checkCuda(cudaMalloc((void**)&d_out_soa, N * sizeof(float)), "cudaMalloc d_out_soa");

    // Initialize input data on device
    // For AoS, we set x=y=z=1.0f for simplicity
    // We can launch a kernel to fill the data
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Fill AoS data
    __global__ void fill_aos(Vec3* d_structs, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            d_structs[idx].x = 1.0f;
            d_structs[idx].y = 1.0f;
            d_structs[idx].z = 1.0f;
        }
    }
    fill_aos<<<blocksPerGrid, threadsPerBlock>>>(d_structs, N);
    checkCuda(cudaDeviceSynchronize(), "fill_aos kernel");

    // Fill SoA data
    __global__ void fill_soa(float* d_x, float* d_y, float* d_z, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            d_x[idx] = 1.0f;
            d_y[idx] = 1.0f;
            d_z[idx] = 1.0f;
        }
    }
    fill_soa<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, N);
    checkCuda(cudaDeviceSynchronize(), "fill_soa kernel");

    // Timing variables
    cudaEvent_t start, stop;
    float time_aos = 0.0f;
    float time_soa = 0.0f;

    // Measure AoS kernel
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    checkCuda(cudaEventRecord(start), "cudaEventRecord start");
    kernel_aos<<<blocksPerGrid, threadsPerBlock>>>(d_structs, d_out_aos, N);
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    checkCuda(cudaEventElapsedTime(&time_aos, start, stop), "cudaEventElapsedTime");
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop");

    // Measure SoA kernel
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    checkCuda(cudaEventRecord(start), "cudaEventRecord start");
    kernel_soa<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, d_out_soa, N);
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    checkCuda(cudaEventElapsedTime(&time_soa, start, stop), "cudaEventElapsedTime");
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop");

    // Copy results back
    checkCuda(cudaMemcpy(h_out_aos, d_out_aos, N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy h_out_aos");
    checkCuda(cudaMemcpy(h_out_soa, d_out_soa, N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy h_out_soa");

    // Verify results (should all be 3.0f)
    bool valid_aos = true;
    bool valid_soa = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_out_aos[i] - 3.0f) > 1e-5f) { valid_aos = false; break; }
    }
    for (int i = 0; i < N; ++i) {
        if (fabs(h_out_soa[i] - 3.0f) > 1e-5f) { valid_soa = false; break; }
    }

    printf("AoS kernel time: %f ms, Result valid: %s\n", time_aos, valid_aos ? "yes" : "no");
    printf("SoA kernel time: %f ms, Result valid: %s\n", time_soa, valid_soa ? "yes" : "no");
    printf("Performance ratio (AoS / SoA): %f\n", time_aos / time_soa);

    // Clean up
    checkCuda(cudaFree(d_structs), "cudaFree d_structs");
    checkCuda(cudaFree(d_out_aos), "cudaFree d_out_aos");
    checkCuda(cudaFree(d_x), "cudaFree d_x");
    checkCuda(cudaFree(d_y), "cudaFree d_y");
    checkCuda(cudaFree(d_z), "cudaFree d_z");
    checkCuda(cudaFree(d_out_soa), "cudaFree d_out_soa");
    free(h_out_aos);
    free(h_out_soa);

    return 0;
}
```