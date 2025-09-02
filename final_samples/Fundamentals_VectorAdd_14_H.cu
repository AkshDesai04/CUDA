```cuda
/*
Aim of the program:
Check if your GPU supports native half-precision (`__half`) arithmetic and try to implement vector add with it.

Thinking:
- CUDA supports native half-precision arithmetic on GPUs with compute capability 5.3 (Volta) or higher.
  Therefore, at runtime we query the device properties (`cudaGetDeviceProperties`) and
  determine if the current GPU is 5.3+ (`major==5 && minor>=3` or `major>=6`).
- Two kernels are provided:
  * `vectorAddHalf` operates on `__half` arrays and uses the intrinsic
    `__half2float` / `__float2half` for conversions.
  * `vectorAddFloat` operates on `float` arrays and is the fallback.
- The host code allocates pinned host memory for input vectors. The input values
  are initially generated as floats. If half support is available, we convert
  those floats to `__half` on the host before copying them to the device.
- The program then launches the appropriate kernel based on the runtime
  capability detection, copies the result back, and verifies correctness by
  comparing a few elements to the expected sum (converted back to float).
- For simplicity and portability, no external libraries (e.g., Thrust) are
  used; all allocations are done with `cudaMalloc` and `cudaMallocHost`.
- The program prints whether half arithmetic is supported, which kernel
  was launched, and whether the result passed the verification.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define N (1 << 20)  // 1M elements

// Kernel performing vector addition with half precision
__global__ void vectorAddHalf(const __half* a, const __half* b, __half* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // __hadd intrinsic for half addition
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// Kernel performing vector addition with single precision
__global__ void vectorAddFloat(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// Helper function to check for CUDA errors
void checkCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Determine GPU capability
    cudaDeviceProp prop;
    int device = 0;
    checkCuda(cudaGetDevice(&device), "getting current device");
    checkCuda(cudaGetDeviceProperties(&prop, device), "getting device properties");
    bool halfSupported = (prop.major > 5) || (prop.major == 5 && prop.minor >= 3);

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Native half-precision supported: %s\n", halfSupported ? "YES" : "NO");

    // Allocate pinned host memory for input vectors (float)
    float* h_a_float = (float*)malloc(N * sizeof(float));
    float* h_b_float = (float*)malloc(N * sizeof(float));
    float* h_c_float = (float*)malloc(N * sizeof(float)); // for verification

    // Initialize input data
    for (int i = 0; i < N; ++i)
    {
        h_a_float[i] = static_cast<float>(i) * 0.001f;
        h_b_float[i] = static_cast<float>(i) * 0.002f;
    }

    // Device pointers
    void *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    if (halfSupported)
    {
        // Allocate device memory for half
        size_t halfSize = N * sizeof(__half);
        checkCuda(cudaMalloc(&d_a, halfSize), "cudaMalloc d_a");
        checkCuda(cudaMalloc(&d_b, halfSize), "cudaMalloc d_b");
        checkCuda(cudaMalloc(&d_c, halfSize), "cudaMalloc d_c");

        // Convert host floats to __half on the host
        __half* h_a_half = (__half*)malloc(halfSize);
        __half* h_b_half = (__half*)malloc(halfSize);

        for (int i = 0; i < N; ++i)
        {
            h_a_half[i] = __float2half(h_a_float[i]);
            h_b_half[i] = __float2half(h_b_float[i]);
        }

        // Copy to device
        checkCuda(cudaMemcpy(d_a, h_a_half, halfSize, cudaMemcpyHostToDevice), "cudaMemcpy h_a to d_a");
        checkCuda(cudaMemcpy(d_b, h_b_half, halfSize, cudaMemcpyHostToDevice), "cudaMemcpy h_b to d_b");

        // Launch half kernel
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        vectorAddHalf<<<blocks, threadsPerBlock>>>((const __half*)d_a, (const __half*)d_b, (__half*)d_c, N);
        checkCuda(cudaDeviceSynchronize(), "kernel launch");

        // Copy result back
        __half* h_c_half = (__half*)malloc(halfSize);
        checkCuda(cudaMemcpy(h_c_half, d_c, halfSize, cudaMemcpyDeviceToHost), "cudaMemcpy d_c to h_c");

        // Convert back to float for verification
        for (int i = 0; i < N; ++i)
        {
            h_c_float[i] = __half2float(h_c_half[i]);
        }

        // Verify a few elements
        bool ok = true;
        for (int i = 0; i < 10; ++i)
        {
            float expected = h_a_float[i] + h_b_float[i];
            if (fabs(h_c_float[i] - expected) > 1e-3f)
            {
                ok = false;
                printf("Mismatch at %d: got %f, expected %f\n", i, h_c_float[i], expected);
                break;
            }
        }
        printf("Half-precision addition %s\n", ok ? "PASSED" : "FAILED");

        free(h_a_half);
        free(h_b_half);
        free(h_c_half);
    }
    else
    {
        // Allocate device memory for float
        size_t floatSize = N * sizeof(float);
        checkCuda(cudaMalloc(&d_a, floatSize), "cudaMalloc d_a");
        checkCuda(cudaMalloc(&d_b, floatSize), "cudaMalloc d_b");
        checkCuda(cudaMalloc(&d_c, floatSize), "cudaMalloc d_c");

        // Copy to device
        checkCuda(cudaMemcpy(d_a, h_a_float, floatSize, cudaMemcpyHostToDevice), "cudaMemcpy h_a to d_a");
        checkCuda(cudaMemcpy(d_b, h_b_float, floatSize, cudaMemcpyHostToDevice), "cudaMemcpy h_b to d_b");

        // Launch float kernel
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        vectorAddFloat<<<blocks, threadsPerBlock>>>((const float*)d_a, (const float*)d_b, (float*)d_c, N);
        checkCuda(cudaDeviceSynchronize(), "kernel launch");

        // Copy result back
        checkCuda(cudaMemcpy(h_c_float, d_c, floatSize, cudaMemcpyDeviceToHost), "cudaMemcpy d_c to h_c");

        // Verify a few elements
        bool ok = true;
        for (int i = 0; i < 10; ++i)
        {
            float expected = h_a_float[i] + h_b_float[i];
            if (fabs(h_c_float[i] - expected) > 1e-6f)
            {
                ok = false;
                printf("Mismatch at %d: got %f, expected %f\n", i, h_c_float[i], expected);
                break;
            }
        }
        printf("Float addition %s\n", ok ? "PASSED" : "FAILED");
    }

    // Clean up
    free(h_a_float);
    free(h_b_float);
    free(h_c_float);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```