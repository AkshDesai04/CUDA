```cuda
/*
Aim of the program: Implement using __half precision numbers if your GPU supports it.

Thought process:
- The goal is to demonstrate vector addition using half-precision arithmetic when the device supports native __half operations.
- Half-precision support requires a compute capability of at least 5.3 (SM 5.3). We will check the running device's capability at runtime.
- Two kernels are provided: one operating on __half, the other on float. The program will decide which kernel to launch based on device capability.
- Host-side code allocates float arrays, initializes them, converts to __half when needed, and performs device-to-host transfers.
- For host conversion between float and __half we use the helper functions __float2half and __half2float from <cuda_fp16.h>.
- Device kernels are conditionally compiled for architectures >= 5.3 to avoid compilation errors on older GPUs. The half kernel is wrapped in a compile-time guard.
- The program outputs a few sample results to verify correctness.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

// Vector size
const int N = 1 << 20;

// Forward declaration of kernels
__global__ void vectorAddFloat(const float* a, const float* b, float* c, int n);

#if __CUDA_ARCH__ >= 530
__global__ void vectorAddHalf(const __half* a, const __half* b, __half* c, int n);
#endif

// Utility macro for checking CUDA errors
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line "     \
                      << __LINE__ << ": " << cudaGetErrorString(err) << ".\n"; \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

int main()
{
    // Choose device 0
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";

    // Determine if device supports native half arithmetic (>= 5.3)
    bool supportHalf = (prop.major > 5) || (prop.major == 5 && prop.minor >= 3);
    if (supportHalf)
        std::cout << "Half-precision arithmetic supported.\n";
    else
        std::cout << "Half-precision arithmetic NOT supported. Falling back to float.\n";

    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N]; // result

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i) * 0.01f;
        h_b[i] = static_cast<float>(N - i) * 0.01f;
    }

    if (supportHalf) {
        // Allocate device memory for half
        __half *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(__half)));

        // Convert host data to half and copy to device
        __half *h_a_half = new __half[N];
        __half *h_b_half = new __half[N];
        for (int i = 0; i < N; ++i) {
            h_a_half[i] = __float2half(h_a[i]);
            h_b_half[i] = __float2half(h_b[i]);
        }

        CUDA_CHECK(cudaMemcpy(d_a, h_a_half, N * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b_half, N * sizeof(__half), cudaMemcpyHostToDevice));

        // Launch half kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vectorAddHalf<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back and convert to float
        __half *h_c_half = new __half[N];
        CUDA_CHECK(cudaMemcpy(h_c_half, d_c, N * sizeof(__half), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i) {
            h_c[i] = __half2float(h_c_half[i]);
        }

        // Cleanup
        delete[] h_a_half;
        delete[] h_b_half;
        delete[] h_c_half;
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    } else {
        // Allocate device memory for float
        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

        // Copy host data to device
        CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

        // Launch float kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vectorAddFloat<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back
        CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Cleanup
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }

    // Print some sample results
    std::cout << "Sample results (index: a + b = c):\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << i << ": " << h_a[i] << " + " << h_b[i]
                  << " = " << h_c[i] << "\n";
    }

    // Clean up host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}

// Float kernel definition
__global__ void vectorAddFloat(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

#if __CUDA_ARCH__ >= 530
// Half kernel definition
__global__ void vectorAddHalf(const __half* a, const __half* b, __half* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Perform addition using half precision
        c[idx] = __hadd(a[idx], b[idx]);
    }
}
#endif
```