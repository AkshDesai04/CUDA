```cpp
/*
Use `fmaxf()` from the CUDA math library instead of the ternary operator.

The goal of this program is to demonstrate how to replace a ternary operator
used for computing the element-wise maximum of two float arrays with the
CUDA math function `fmaxf()`.  The program performs the following steps:
1. Allocate two input arrays and one output array on the host.
2. Initialize the input arrays with example data.
3. Allocate corresponding memory on the device.
4. Copy the input arrays from host to device.
5. Launch a CUDA kernel that computes `c[i] = fmaxf(a[i], b[i])` for each
   element in parallel.
6. Copy the result array back from device to host.
7. Print the input arrays and the resulting maximum array to verify correctness.
8. Clean up allocated memory.

Using `fmaxf()` instead of the ternary operator (`a[i] > b[i] ? a[i] : b[i]`)
has the advantage of being a built-in function that may be optimized by
the CUDA compiler and hardware, ensuring consistent and potentially faster
execution.  It also improves code readability by making the intent explicit.

The code includes basic CUDA error checking to aid debugging and ensure
proper resource management.  It is self‑contained and can be compiled with
`nvcc`:

    nvcc -o elementwise_max elementwise_max.cu

Author: OpenAI ChatGPT
*/
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel that computes element-wise maximum using fmaxf
__global__ void elementwiseMax(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fmaxf(a[idx], b[idx]); // Using fmaxf instead of ternary
    }
}

int main() {
    const int N = 10;
    size_t size = N * sizeof(float);

    // Allocate host memory
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);          // 0.0, 1.0, 2.0, ...
        h_b[i] = static_cast<float>(N - i - 1);  // 9.0, 8.0, 7.0, ...
    }

    // Allocate device memory
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA(cudaMalloc((void**)&d_c, size));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    elementwiseMax<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Array A: ");
    for (int i = 0; i < N; ++i) printf("%5.1f ", h_a[i]);
    printf("\n");

    printf("Array B: ");
    for (int i = 0; i < N; ++i) printf("%5.1f ", h_b[i]);
    printf("\n");

    printf("Element-wise max (using fmaxf): ");
    for (int i = 0; i < N; ++i) printf("%5.1f ", h_c[i]);
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}
```