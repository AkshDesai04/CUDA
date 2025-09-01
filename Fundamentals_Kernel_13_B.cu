/*
Write a kernel that takes three pointers, `a`, `b`, and `c`, and computes `c[i] = a[i] + b[i]`.

Thinking:
- The goal is to implement a simple element‑wise addition kernel for CUDA.  
- We need to define a `__global__` function that receives three device pointers and the size of the vectors.  
- Inside the kernel, compute a global index from `blockIdx`, `blockDim`, and `threadIdx`.  
- Guard against out‑of‑bounds accesses (`if (idx < N)`).  
- In the host code, allocate host arrays, initialize them, copy to device, launch the kernel, copy the result back, and verify.  
- Use a block size of 256 and a grid size that covers the whole vector.  
- Add basic error‑checking macro to simplify CUDA API calls.  
- Compile with `nvcc` and run to confirm correct behavior.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that performs c[i] = a[i] + b[i]
__global__ void addVectors(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;                 // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    // Copy host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify a few elements
    bool success = true;
    for (int i = 0; i < 10; ++i)
    {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f)
        {
            printf("Mismatch at index %d: host=%f device=%f expected=%f\n",
                   i, h_c[i], expected, h_c[i]);
            success = false;
            break;
        }
    }
    if (success)
        printf("Success! First 10 results verified.\n");

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
