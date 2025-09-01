```cpp
/*
Aim of the program: Implement C[i] = A[i] - scalar.

Thinking and design:

1. Goal is to create a simple CUDA program that performs element-wise subtraction of a scalar value from each element of an array A, producing array C.
2. Use single precision float arrays for demonstration. The scalar is also a float.
3. Host side:
   - Allocate and initialize input array A on host.
   - Allocate output array C on host (optional, used to copy back from device).
   - Allocate device memory for A and C.
   - Copy A to device.
4. Device side:
   - Define a kernel `subtractScalar` that takes pointers to device arrays A and C, the scalar, and array length N.
   - Each thread computes a single element `C[idx] = A[idx] - scalar` if `idx < N` to guard against out-of-bounds.
   - Use a simple 1D grid of 1D blocks; compute grid size with `(N + threadsPerBlock - 1)/threadsPerBlock`.
5. After kernel launch, copy C back to host.
6. Print a few values to verify correctness.
7. Clean up device memory and return.
8. Include basic error checking via a macro `checkCuda` to capture CUDA API errors.
9. The program is self‑contained, compiles with `nvcc`, and can be run on any GPU that supports the CUDA Runtime API.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro for error checking
#define checkCuda(call)                                                     \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),        \
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel to subtract scalar from each element of array A
__global__ void subtractScalar(const float *A, float *C, float scalar, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] - scalar;
    }
}

int main() {
    const int N = 1 << 20;            // 1M elements
    const float scalar = 5.0f;

    // Allocate host memory
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));
    if (!h_A || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host input array A
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_A = nullptr;
    float *d_C = nullptr;
    checkCuda(cudaMalloc((void**)&d_A, N * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Copy input array to device
    checkCuda(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));

    // Define execution configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    subtractScalar<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, scalar, N);
    checkCuda(cudaGetLastError());   // Check for launch errors

    // Copy result back to host
    checkCuda(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("First 10 results (A[i] - %f):\n", scalar);
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up
    free(h_A);
    free(h_C);
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_C));

    return EXIT_SUCCESS;
}
```