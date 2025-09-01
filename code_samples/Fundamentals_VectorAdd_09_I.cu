/*
Aim of the program: Calculate the dot product of two vectors (this is a reduction, a more advanced topic, but can be done naively with atomics).

Thinking process:
1. We need a CUDA kernel that multiplies corresponding elements of two arrays and accumulates the result into a single scalar.
2. A straightforward reduction would use shared memory and a hierarchical sum, but the prompt asks for a naive approach using atomics.
3. CUDA provides atomicAdd for float types on devices of compute capability >= 2.0, which we will use to accumulate each thread's partial product into a global result variable.
4. The kernel will be launched with enough threads to cover all elements. Each thread will:
   - Compute its global index.
   - If the index is within bounds, multiply a[idx] * b[idx].
   - Atomically add the product to the global result pointer.
5. Host code:
   - Allocate and initialize host vectors with some test values.
   - Allocate device memory and copy the vectors.
   - Allocate device memory for the result and set it to 0.
   - Launch the kernel with appropriate grid and block dimensions.
   - Copy the result back to the host and print it.
6. Include basic error checking macros for CUDA API calls.
7. The code is written for a single-file .cu program that can be compiled with nvcc.
*/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

__global__ void dotProductKernel(const float *a, const float *b, float *res, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float prod = a[idx] * b[idx];
        atomicAdd(res, prod);
    }
}

int main()
{
    const int N = 1 << 20; // 1,048,576 elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    if (!h_a || !h_b) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors with random data
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_res;
    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_res, sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Initialize result to 0
    CHECK_CUDA(cudaMemset(d_res, 0, sizeof(float)));

    // Launch kernel
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    dotProductKernel<<<gridSize, blockSize>>>(d_a, d_b, d_res, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    float h_res = 0.0f;
    CHECK_CUDA(cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Dot product result: %f\n", h_res);

    // Cleanup
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_res));
    free(h_a);
    free(h_b);

    return 0;
}
