/*
Aim: Calculate the dot product of two vectors (this is a reduction, a more advanced topic, but can be done naively with atomics).

Thinking process:
- The dot product requires multiplying corresponding elements of two arrays and summing the products.
- CUDA kernels run in parallel; each thread can handle one element pair.
- We can let each thread compute a product and then use an atomic add to a global accumulator.
- This naive approach is simple but can suffer from atomic contention when many threads try to update the same memory location concurrently.
- For demonstration purposes and small to medium vector sizes, the atomic approach is acceptable.
- We choose float type for simplicity; atomicAdd on float is available on compute capability 2.0+.
- Steps:
  1. Allocate host vectors, fill them with sample values.
  2. Allocate device vectors and copy data.
  3. Allocate device result variable and initialize to 0.
  4. Launch kernel with a suitable grid/block configuration.
  5. Copy result back to host and print.
  6. Verify against a CPU reference implementation.
- Error checking for CUDA API calls is included for robustness.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel that computes element-wise product and atomically adds to global sum
__global__ void dotProductKernel(const float *a, const float *b, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float prod = a[idx] * b[idx];
        atomicAdd(result, prod);
    }
}

// Simple CPU reference implementation for verification
float dotProductCPU(const float *a, const float *b, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

// CUDA error checking helper
void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    const int N = 1 << 20; // 1,048,576 elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    if (!h_a || !h_b)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = 1.0f; // or any pattern
        h_b[i] = 2.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_result;
    checkCuda(cudaMalloc((void**)&d_a, size), "cudaMalloc d_a");
    checkCuda(cudaMalloc((void**)&d_b, size), "cudaMalloc d_b");
    checkCuda(cudaMalloc((void**)&d_result, sizeof(float)), "cudaMalloc d_result");

    // Copy data to device
    checkCuda(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "Memcpy h_a to d_a");
    checkCuda(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "Memcpy h_b to d_b");

    // Initialize result on device to zero
    float zero = 0.0f;
    checkCuda(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice), "Memcpy zero to d_result");

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, N);
    checkCuda(cudaGetLastError(), "Kernel launch");

    // Copy result back to host
    float h_result = 0.0f;
    checkCuda(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost), "Memcpy d_result to h_result");

    // Compute reference result
    float cpu_result = dotProductCPU(h_a, h_b, N);

    // Print results
    printf("GPU dot product result: %.6f\n", h_result);
    printf("CPU dot product result: %.6f\n", cpu_result);
    printf("Difference: %.6e\n", fabsf(h_result - cpu_result));

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(h_a);
    free(h_b);

    return 0;
}
