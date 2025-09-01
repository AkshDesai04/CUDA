```cpp
/*
Perform the vector addition in-place: `A[i] = A[i] + B[i]`.

The aim of this program is to demonstrate a simple CUDA kernel that performs an element-wise addition of two vectors
in-place on the device.  The kernel will read the ith element of vector A and vector B, compute the sum, and store the
result back into the same position of vector A.  The program will:
1. Allocate two host vectors A and B and fill them with sample data.
2. Allocate corresponding device vectors dA and dB.
3. Copy the host data to the device.
4. Launch a CUDA kernel that does `A[i] = A[i] + B[i]` for all i.
5. Copy the updated vector A back to the host.
6. Verify the result and clean up memory.

The kernel uses a simple 1D grid of 1D blocks.  The block size is chosen to be a power of two (256 threads per block)
for efficient utilization.  Error checking is performed after each CUDA API call via a macro to simplify debugging.

The program is self‑contained and can be compiled with `nvcc vector_add_inplace.cu -o vector_add_inplace` and run
directly.  The vector size is set to one million elements (change N if desired).  The output shows the first few
entries before and after the addition to confirm correctness.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

// Kernel performing in-place vector addition: A[i] = A[i] + B[i]
__global__ void vectorAddInPlace(float* A, const float* B, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        A[idx] += B[idx];
    }
}

int main(void)
{
    const size_t N = 1 << 20; // 1,048,576 elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    if (!h_A || !h_B) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);          // A[i] = i
        h_B[i] = static_cast<float>(2 * i);      // B[i] = 2*i
    }

    // Allocate device memory
    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddInPlace<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("First 5 elements after addition:\n");
    for (int i = 0; i < 5; ++i) {
        printf("A[%d] = %f (expected %f)\n", i, h_A[i], static_cast<float>(i) + static_cast<float>(2 * i));
    }

    // Clean up
    free(h_A);
    free(h_B);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    return 0;
}
```