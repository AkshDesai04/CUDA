```cuda
/*
Aim of the program: Pass the scalar via constant memory.

Thinking process:
1. The goal is to demonstrate how a scalar value can be stored in CUDA's constant memory space
   and accessed efficiently by all threads during a kernel launch. Constant memory is read-only
   from the perspective of the kernel and is cached, making repeated accesses from many threads
   very fast.

2. To keep the example simple yet illustrative, we will perform a vector multiply operation:
   y[i] = x[i] * scalar. The scalar will be stored in constant memory. The input vector x
   will be stored in global memory, as will the output vector y.

3. Steps:
   a. Declare a constant memory variable `__constant__ float d_scalar;` at the device side.
   b. On the host, initialize a scalar value (e.g., 2.5f) and copy it to the constant memory
      using `cudaMemcpyToSymbol`.
   c. Allocate and initialize input data on the host, then copy it to the device.
   d. Launch a kernel that reads the scalar from constant memory and multiplies each element.
   e. Copy the result back to host and verify correctness.
   f. Use simple error checking wrappers for CUDA calls.

4. Performance note:
   Because the scalar is constant for all threads, the GPU will load it once into the L1 cache
   for the constant address space and serve it from cache for all threads, providing maximum
   throughput for this type of small constant value.

5. The code below is self-contained, includes basic error handling, and can be compiled with
   `nvcc -arch=sm_52 -o const_mem_example const_mem_example.cu` (or any other supported
   architecture). The example will run on the default device.

6. The main focus is on demonstrating the use of `__constant__` memory and `cudaMemcpyToSymbol`,
   not on optimizing vector multiply beyond the straightforward approach.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Define constant memory for the scalar
__constant__ float d_scalar;

// Kernel that multiplies each element of an array by the scalar stored in constant memory
__global__ void vecMulKernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] * d_scalar;
    }
}

// Simple CUDA error checking macro
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err__ = (err); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    const int N = 1 << 20;  // 1M elements
    const int bytes = N * sizeof(float);

    // Allocate host memory
    float* h_x = (float*)malloc(bytes);
    float* h_y = (float*)malloc(bytes);
    if (!h_x || !h_y) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i) * 0.01f;  // arbitrary values
    }

    // Device memory allocation
    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_x, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_y, bytes));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    // Set scalar value in constant memory
    const float h_scalar = 2.5f;
    CUDA_CHECK(cudaMemcpyToSymbol(d_scalar, &h_scalar, sizeof(float)));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    // Verify results (simple check on first 10 elements)
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_x[i] * h_scalar;
        if (fabs(h_y[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: host %f, device %f, expected %f\n",
                    i, h_y[i], expected, expected);
            success = false;
            break;
        }
    }
    if (success) {
        printf("First 10 results verified successfully.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);

    return EXIT_SUCCESS;
}
```