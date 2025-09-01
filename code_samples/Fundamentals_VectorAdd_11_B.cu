```cpp
/*
Launch your grid-stride kernel for a vector of size 1M, but only launch a grid of 1024 threads total. Verify the correctness.

Thinking:
- Need a CUDA .cu file that demonstrates a grid-stride loop.
- Vector size N = 1,000,000 (1M).
- Total threads launched must be 1024, so use a single block of 1024 threads.
- The kernel will increment each element by 1 using a grid-stride pattern.
- After kernel execution, copy data back to host and verify that each element was incremented correctly.
- Include simple error-checking macro for CUDA calls.
- Use standard C++ (no external libraries beyond CUDA runtime).
- The code should compile with `nvcc` and run without additional input.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

// Kernel that increments each element by 1 using grid-stride loop
__global__ void increment_kernel(int *data, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < N; i += stride)
    {
        data[i] += 1;
    }
}

int main(void)
{
    const size_t N = 1'000'000; // 1M elements
    const size_t bytes = N * sizeof(int);

    // Allocate host memory
    int *h_in = (int *)malloc(bytes);
    int *h_out = (int *)malloc(bytes);
    if (!h_in || !h_out)
    {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with known values
    for (size_t i = 0; i < N; ++i)
    {
        h_in[i] = static_cast<int>(i % 100); // small pattern
    }

    // Allocate device memory
    int *d_data = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_data, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_in, bytes, cudaMemcpyHostToDevice));

    // Launch kernel with 1024 total threads: 1 block of 1024 threads
    const int threads_per_block = 1024;
    const int blocks = 1; // total threads = threads_per_block * blocks = 1024
    increment_kernel<<<blocks, threads_per_block>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_data, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness
    bool correct = true;
    for (size_t i = 0; i < N; ++i)
    {
        int expected = h_in[i] + 1;
        if (h_out[i] != expected)
        {
            fprintf(stderr, "Mismatch at index %zu: expected %d, got %d\n",
                    i, expected, h_out[i]);
            correct = false;
            break;
        }
    }

    if (correct)
    {
        printf("Verification succeeded: all elements incremented correctly.\n");
    }
    else
    {
        printf("Verification failed.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    free(h_in);
    free(h_out);

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
```