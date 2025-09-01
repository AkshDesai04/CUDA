```cpp
/*
Use `float`s for the vectors instead of `int`s.

Thinking:
1. We will implement a simple CUDA vector addition program where two input arrays of floats are added element-wise and stored in an output array.
2. We allocate memory on device using cudaMalloc for float arrays.
3. The kernel will compute the global index using blockIdx.x * blockDim.x + threadIdx.x, and if the index is within bounds, it will compute the sum of the corresponding elements from the two input arrays.
4. On the host side we set up N elements, initialize host arrays with some values (e.g., 1.0f, 2.0f), copy them to the device, launch the kernel, copy the result back, print a few results, and then free all allocated memory.
5. We will use float precision; we include error checking after each CUDA API call to ensure any issues are reported.
6. The grid and block dimensions are chosen to be 256 threads per block; the grid size is computed as (N + blockSize - 1) / blockSize to cover all elements.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                               \
    {                                                                  \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    }

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {
    const int N = 1 << 20;  // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_c, bytes));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify result (print first 10 elements)
    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_c[%d] = %f\n", i, h_c[i]);
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}
```