```cuda
/*
Aim: Check if your GPU supports native half-precision (`__half`) arithmetic and try to implement vector add with it.

Thinking:
1. Native half-precision arithmetic is available on GPUs with compute capability 5.3 and above.
2. To determine support we query the device's compute capability using cudaDeviceGetAttribute for
   CUDA_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR and CUDA_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR.
3. If the GPU does not meet the required compute capability we print an informative message and exit.
4. For demonstration we create two float vectors on the host, convert them to half using
   __float2half, copy them to device, run a vector addition kernel that operates on __half
   values, copy the result back, convert it to float, and display a few elements.
5. The kernel uses __hadd to perform half addition.
6. We include <cuda_fp16.h> to get the half types and conversion intrinsics.
7. The code is self-contained, compiles with nvcc targeting a compute capability of 5.3 or higher,
   and prints whether the GPU supports native half-precision arithmetic.

*/

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel that adds two vectors of __half and stores the result in C
__global__ void vectorAddHalf(const __half* A, const __half* B, __half* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // Use native half addition
        C[idx] = __hadd(A[idx], B[idx]);
    }
}

int main()
{
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    // Determine compute capability
    int major = prop.major;
    int minor = prop.minor;

    bool halfSupported = (major > 5) || (major == 5 && minor >= 3);

    printf("Device %d: %s (compute %d.%d)\n",
           device, prop.name, major, minor);

    if (!halfSupported)
    {
        printf("This GPU does not support native half-precision arithmetic (requires compute capability 5.3+).\n");
        return 0;
    }

    // Parameters
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Host memory allocation
    float *h_A_f = (float*)malloc(size);
    float *h_B_f = (float*)malloc(size);
    float *h_C_f = (float*)malloc(size);

    // Initialize host arrays with some values
    for (int i = 0; i < N; ++i)
    {
        h_A_f[i] = (float)i * 0.001f;
        h_B_f[i] = (float)(N - i) * 0.001f;
    }

    // Convert host float arrays to half
    __half *h_A_h = (__half*)malloc(N * sizeof(__half));
    __half *h_B_h = (__half*)malloc(N * sizeof(__half));
    for (int i = 0; i < N; ++i)
    {
        h_A_h[i] = __float2half(h_A_f[i]);
        h_B_h[i] = __float2half(h_B_f[i]);
    }

    // Device memory allocation
    __half *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(__half));
    cudaMalloc((void**)&d_B, N * sizeof(__half));
    cudaMalloc((void**)&d_C, N * sizeof(__half));

    // Copy inputs to device
    cudaMemcpy(d_A, h_A_h, N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_h, N * sizeof(__half), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddHalf<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for completion
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C_f, d_C, N * sizeof(__half), cudaMemcpyDeviceToHost);

    // Convert result from half to float for display
    for (int i = 0; i < N; ++i)
    {
        h_C_f[i] = __half2float(h_C_f[i]); // reinterpret as half then back to float
    }

    // Display a few results
    printf("\nSample results (index : A + B):\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("%d: %.6f + %.6f = %.6f\n",
               i, h_A_f[i], h_B_f[i], h_C_f[i]);
    }

    // Clean up
    free(h_A_f);
    free(h_B_f);
    free(h_C_f);
    free(h_A_h);
    free(h_B_h);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```