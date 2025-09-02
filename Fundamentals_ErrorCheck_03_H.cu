```cpp
/* 
In a program with multiple streams, `cudaDeviceSynchronize` is a heavyweight operation. What is a more fine-grained alternative? (`cudaStreamSynchronize`).

Thinking: 
- The prompt asks for a more fine-grained alternative to `cudaDeviceSynchronize`, which is `cudaStreamSynchronize`. 
- To demonstrate this, I'll write a small CUDA C program that creates multiple streams, launches a simple kernel on each stream, and then synchronizes each stream individually using `cudaStreamSynchronize`. 
- This avoids a global device-wide sync and allows work in other streams or the host to continue while waiting for a specific stream to finish. 
- The code will include basic error checking, host and device memory allocation, asynchronous memory copies, kernel launches, stream synchronization, and cleanup. 
- The aim is to illustrate that `cudaStreamSynchronize` can be used for fine-grained synchronization, providing a more efficient alternative to `cudaDeviceSynchronize`. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that adds two vectors
__global__ void addKernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// Error checking macro
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    __func__, __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Host memory allocation
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Device memory allocation
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA(cudaMalloc((void**)&d_c, size));

    // Create two streams
    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    // Asynchronously copy data to device in stream1
    CHECK_CUDA(cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1));
    // Asynchronously copy data to device in stream2
    CHECK_CUDA(cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream2));

    // Launch kernel on stream1
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());

    // Launch kernel on stream2
    addKernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_b, d_a, d_c, N);
    CHECK_CUDA(cudaGetLastError());

    // Synchronize streams individually
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, 0));

    // Wait for all operations to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify result (optional)
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (abs(h_c[i] - expected) > 1e-5f) {
            if (errors < 10) {
                fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n",
                        i, h_c[i], expected);
            }
            ++errors;
        }
    }
    if (errors == 0) {
        printf("All results match expected values.\n");
    } else {
        printf("Total mismatches: %d\n", errors);
    }

    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```