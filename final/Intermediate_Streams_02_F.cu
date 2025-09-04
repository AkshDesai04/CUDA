```c
/*
Remember to destroy the streams with `cudaStreamDestroy` at the end.

Thought process:
- The user wants a CUDA program that demonstrates the use of streams and reminds to clean them up.
- Since no specific functionality was given, a classic vector addition example using two streams is suitable.
- The code will:
  1. Allocate host and device memory.
  2. Create two CUDA streams.
  3. Asynchronously copy input data to device on each stream.
  4. Launch a simple kernel on each stream to perform addition.
  5. Asynchronously copy results back to host on each stream.
  6. Synchronize the streams to ensure completion.
  7. Destroy both streams with cudaStreamDestroy.
- This fulfills the requirement to use and then destroy streams.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main(void) {
    size_t size = N * sizeof(float);
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 1.0f;
        h_b[i] = (N - i) * 1.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Asynchronously copy data to device on stream1
    cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream1);

    // Asynchronously copy data to device on stream2
    cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream2);

    // Launch kernels on both streams
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_a, d_b, d_c, N);
    vecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_a, d_b, d_c, N);

    // Asynchronously copy result back to host on each stream
    cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream2);

    // Wait for streams to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Validate results
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (abs(h_c[i] - expected) > 1e-5f) {
            ++errors;
            if (errors <= 10) {
                printf("Mismatch at %d: got %f, expected %f\n", i, h_c[i], expected);
            }
        }
    }
    if (errors == 0) {
        printf("All results are correct.\n");
    } else {
        printf("Found %d mismatches.\n", errors);
    }

    // Destroy streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```