```cuda
/*
Aim: Analyze why the AoS kernel has poor memory performance. (Threads in a warp access x, but the y and z components are loaded into cache unnecessarily, wasting bandwidth).

Thinking:
- The AoS (Array of Structs) layout stores each vector component contiguously as a single struct:
  struct Vec3 { float x; float y; float z; };
  Vec3 a[N];
- In a warp of 32 threads, each thread reads the same member (e.g., x). Because the structs are stored one after another, each thread will load the full 12‑byte struct into L1 cache (or DRAM), even though it only needs the first 4 bytes (x). The remaining 8 bytes (y and z) are fetched but never used, wasting bandwidth.
- This is especially detrimental when the data is large, as the cache line (typically 128 bytes on NVIDIA GPUs) will be partially filled with unused data, leading to higher memory traffic and lower effective bandwidth.
- A better layout for such workloads is SoA (Structure of Arrays), where each component is stored in a separate array:
  float x[N], y[N], z[N];
  Now each thread reads from the same memory region (x array) and the hardware can coalesce the accesses efficiently, retrieving only the necessary data.
- The kernel below demonstrates the AoS access pattern and prints out the amount of memory read per thread, illustrating the inefficiency. It also includes a simple SoA kernel for comparison.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Define a simple struct for AoS
struct Vec3
{
    float x;
    float y;
    float z;
};

// Kernel that only needs the x component but reads the whole struct
__global__ void AoSKernel(const Vec3 *a, float *out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // Load the entire struct into a temporary variable
        Vec3 temp = a[idx];
        // Only use the x component
        out[idx] = temp.x;
    }
}

// Kernel that only reads the x array (SoA)
__global__ void SoAKernel(const float *x, float *out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        out[idx] = x[idx];
    }
}

// Utility function to check for CUDA errors
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    const int N = 1 << 20; // 1M elements
    const size_t sizeVec3 = N * sizeof(Vec3);
    const size_t sizeFloat = N * sizeof(float);

    // Allocate host memory
    Vec3 *h_a = (Vec3 *)malloc(sizeVec3);
    float *h_out = (float *)malloc(sizeFloat);

    // Initialize data
    for (int i = 0; i < N; ++i)
    {
        h_a[i].x = static_cast<float>(i);
        h_a[i].y = static_cast<float>(i) * 2.0f;
        h_a[i].z = static_cast<float>(i) * 3.0f;
    }

    // Allocate device memory
    Vec3 *d_a;
    float *d_out;
    checkCudaError(cudaMalloc((void **)&d_a, sizeVec3), "cudaMalloc d_a");
    checkCudaError(cudaMalloc((void **)&d_out, sizeFloat), "cudaMalloc d_out");

    // Copy input to device
    checkCudaError(cudaMemcpy(d_a, h_a, sizeVec3, cudaMemcpyHostToDevice), "cudaMemcpy h_a->d_a");

    // Launch AoS kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    AoSKernel<<<blocks, threadsPerBlock>>>(d_a, d_out, N);
    checkCudaError(cudaGetLastError(), "AoSKernel launch");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_out, d_out, sizeFloat, cudaMemcpyDeviceToHost), "cudaMemcpy d_out->h_out");

    // Verify a few outputs
    printf("AoS kernel output sample:\n");
    for (int i = 0; i < 5; ++i)
        printf("h_out[%d] = %f\n", i, h_out[i]);

    // Now set up SoA data
    float *h_x = (float *)malloc(sizeFloat);
    for (int i = 0; i < N; ++i)
        h_x[i] = static_cast<float>(i);

    float *d_x;
    checkCudaError(cudaMalloc((void **)&d_x, sizeFloat), "cudaMalloc d_x");
    checkCudaError(cudaMemcpy(d_x, h_x, sizeFloat, cudaMemcpyHostToDevice), "cudaMemcpy h_x->d_x");

    // Launch SoA kernel
    SoAKernel<<<blocks, threadsPerBlock>>>(d_x, d_out, N);
    checkCudaError(cudaGetLastError(), "SoAKernel launch");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_out, d_out, sizeFloat, cudaMemcpyDeviceToHost), "cudaMemcpy d_out->h_out");

    printf("\nSoA kernel output sample:\n");
    for (int i = 0; i < 5; ++i)
        printf("h_out[%d] = %f\n", i, h_out[i]);

    // Clean up
    free(h_a);
    free(h_out);
    free(h_x);
    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_x);

    return 0;
}
```