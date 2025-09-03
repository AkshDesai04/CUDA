```cpp
/*
Aim of the program (exactly as requested):
Check if your GPU supports native half-precision (`__half`) arithmetic and try to implement vector add with it.

Thoughts and design decisions:
1. The CUDA runtime API exposes a boolean field `canUseHalfPrecisionArithmetic` in `cudaDeviceProp` that indicates whether the device can execute native half-precision (FP16) instructions.  We query this field for the chosen device and report the capability.  If the device does not support it we exit gracefully.

2. Vector addition is implemented as a kernel that accepts pointers to `__half` arrays.  We use the half-precision arithmetic intrinsic `__hadd` to perform the addition.  This intrinsic requires a compute capability of at least 5.3, which is guaranteed by the check in step 1.

3. Host code works with regular `float` arrays for convenience.  Before launching the kernel we convert the input `float` data to `__half` on the host, copy the half arrays to the device, launch the kernel, and finally copy the result back to the host.  The resulting half array is then converted back to `float` for verification and printing.

4. For simplicity and clarity, error handling is performed via a helper macro `CUDA_CHECK`.  The program allocates memory on the host for both `float` and `__half` representations of the input vectors.  Device memory is allocated for the three half vectors involved in the operation.

5. After the kernel execution, the program verifies the result by comparing it to the expected sum computed on the host in `float`.  A few sample elements are printed to confirm correctness.

6. The code is written to compile with `nvcc` targeting CUDA 11 or newer.  The use of `cuda_fp16.h` and the `__half` type ensures compatibility with devices that support native half arithmetic.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that performs element-wise addition of two half-precision vectors
__global__ void vectorAddHalf(__half* c, const __half* a, const __half* b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);  // Native half-precision addition
    }
}

int main(void)
{
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device %d: %s\n", device, prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    if (!prop.canUseHalfPrecisionArithmetic) {
        printf("This device does not support native half-precision arithmetic.\n");
        return EXIT_FAILURE;
    }
    printf("Device supports native half-precision arithmetic.\n");

    // Vector size
    const int N = 1 << 20;  // 1,048,576 elements
    const size_t bytes = N * sizeof(float);
    const size_t halfBytes = N * sizeof(__half);

    // Allocate host memory for input and output (float)
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    float *h_c_expected = (float*)malloc(bytes);

    // Allocate host memory for half representation
    __half *h_a_half = (__half*)malloc(halfBytes);
    __half *h_b_half = (__half*)malloc(halfBytes);
    __half *h_c_half = (__half*)malloc(halfBytes);

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i * 0.001f;
        h_b[i] = (float)(N - i) * 0.001f;
        h_c_expected[i] = h_a[i] + h_b[i];
    }

    // Convert float inputs to half on host
    for (int i = 0; i < N; ++i) {
        h_a_half[i] = __float2half(h_a[i]);
        h_b_half[i] = __float2half(h_b[i]);
    }

    // Allocate device memory
    __half *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, halfBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, halfBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, halfBytes));

    // Copy half inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a_half, halfBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b_half, halfBytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddHalf<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c_half, d_c, halfBytes, cudaMemcpyDeviceToHost));

    // Convert result to float
    for (int i = 0; i < N; ++i) {
        h_c[i] = __half2float(h_c_half[i]);
    }

    // Verify a few elements
    int passed = 1;
    for (int i = 0; i < 10; ++i) {
        if (fabs(h_c[i] - h_c_expected[i]) > 1e-3f) {
            printf("Mismatch at index %d: expected %f, got %f\n",
                   i, h_c_expected[i], h_c[i]);
            passed = 0;
            break;
        }
    }
    if (passed) {
        printf("Verification passed for first 10 elements.\n");
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_expected);
    free(h_a_half);
    free(h_b_half);
    free(h_c_half);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return EXIT_SUCCESS;
}
```