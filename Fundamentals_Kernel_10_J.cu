```cuda
/*
Pass the struct by constant reference (const MyData&) to the kernel. This can be more efficient for large structs.
Thinking:
- In CUDA, kernel arguments must be trivially copyable types, so passing a reference directly as a kernel parameter
  is not supported. To simulate passing a struct by constant reference, we pass a pointer to the struct and create a
  reference inside the kernel. This allows the kernel to treat the data as a reference while avoiding the overhead of
  copying large structs between host and device.
- We define a large struct `MyData` with many float members to emulate a sizable data structure.
- The host allocates an array of `MyData`, initializes it, and copies it to device memory.
- The kernel receives a pointer to the array, creates a `const MyData&` reference for each element, and performs
  some simple computation (summing all float members) to produce an integer result.
- Results are copied back to host and printed.
- Basic CUDA error checking is included.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>

// Large struct definition
struct MyData {
    float a[32];  // 32 floats -> 128 bytes
    float b[32];
    float c[32];
    float d[32];
    float e[32];
    float f[32];
    float g[32];
    float h[32];
    float i[32];
    float j[32];
};

// Device function that processes a const reference to MyData
__device__ __forceinline__ int processData(const MyData& data)
{
    // Simple computation: sum all members of the struct
    int sum = 0;
    for (int k = 0; k < 32; ++k) sum += static_cast<int>(data.a[k]);
    for (int k = 0; k < 32; ++k) sum += static_cast<int>(data.b[k]);
    for (int k = 0; k < 32; ++k) sum += static_cast<int>(data.c[k]);
    for (int k = 0; k < 32; ++k) sum += static_cast<int>(data.d[k]);
    for (int k = 0; k < 32; ++k) sum += static_cast<int>(data.e[k]);
    for (int k = 0; k < 32; ++k) sum += static_cast<int>(data.f[k]);
    for (int k = 0; k < 32; ++k) sum += static_cast<int>(data.g[k]);
    for (int k = 0; k < 32; ++k) sum += static_cast<int>(data.h[k]);
    for (int k = 0; k < 32; ++k) sum += static_cast<int>(data.i[k]);
    for (int k = 0; k < 32; ++k) sum += static_cast<int>(data.j[k]);
    return sum;
}

// Kernel that receives a pointer to an array of MyData and outputs an integer per element
__global__ void myKernel(const MyData* d_in, int* d_out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Create a const reference to the element
        const MyData& ref = d_in[idx];
        d_out[idx] = processData(ref);
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : " \
                      << cudaGetErrorString(err_) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main()
{
    const int N = 256; // number of elements
    size_t structSize = sizeof(MyData);
    size_t totalSize = N * structSize;

    // Allocate host memory
    MyData* h_in = new MyData[N];
    int* h_out = new int[N];

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 32; ++k) {
            h_in[i].a[k] = static_cast<float>(i + k);
            h_in[i].b[k] = static_cast<float>(i - k);
            h_in[i].c[k] = static_cast<float>(i * k);
            h_in[i].d[k] = static_cast<float>(i / (k + 1));
            h_in[i].e[k] = static_cast<float>(i + k * 2);
            h_in[i].f[k] = static_cast<float>(i - k * 3);
            h_in[i].g[k] = static_cast<float>(i + k * 4);
            h_in[i].h[k] = static_cast<float>(i - k * 5);
            h_in[i].i[k] = static_cast<float>(i + k * 6);
            h_in[i].j[k] = static_cast<float>(i - k * 7);
        }
    }

    // Allocate device memory
    MyData* d_in;
    int* d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, totalSize));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(int)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, totalSize, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print a few results
    std::cout << "Sample results:" << std::endl;
    for (int i = 0; i < std::min(N, 8); ++i) {
        std::cout << "h_out[" << i << "] = " << h_out[i] << std::endl;
    }

    // Clean up
    delete[] h_in;
    delete[] h_out;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
```