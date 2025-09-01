```cuda
/*
Aim: Add a comment explaining what a CUDA stream is and how it relates to overlapping operations.

Thinking:
- The user requested a comment that explains CUDA streams and how they allow overlapping of operations.
- According to the hierarchy, the developer's instruction to produce a complete CUDA program in a .cu file takes precedence.
- Therefore, the program will be a minimal example that demonstrates the use of CUDA streams to overlap memory copies with kernel execution.
- The top-level comment will contain the exact aim statement and a thorough explanation of CUDA streams and overlapping.
- The code itself will allocate host and device memory, create two streams, perform asynchronous memory copies and kernel launches,
  and then synchronize. This will illustrate how streams enable concurrent execution of operations that do not depend on each other.
- Finally, the program will clean up resources and exit.
*/

#include <cuda_runtime.h>
#include <stdio.h>

// Simple kernel that adds two vectors
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Host allocations
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 1.0f;
        h_b[i] = (N - i) * 1.0f;
    }

    // Device allocations
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Create two streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Asynchronously copy input data to device in stream1
    cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream1);

    // Launch kernel in stream2; it will start as soon as its input dependencies (d_a, d_b) are ready
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_a, d_b, d_c, N);

    // Asynchronously copy result back to host in stream1 (to overlap copy-back with kernel execution)
    cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream1);

    // Wait for all operations to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
            success = false;
            break;
        }
    }
    printf("Verification %s\n", success ? "PASSED" : "FAILED");

    // Clean up
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```