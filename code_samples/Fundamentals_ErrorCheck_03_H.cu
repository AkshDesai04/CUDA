```cpp
/*
In a program with multiple streams, `cudaDeviceSynchronize` is a heavyweight operation. What is a more fine-grained alternative? (`cudaStreamSynchronize`).

The goal of this example program is to demonstrate why cudaDeviceSynchronize is considered heavyweight and how cudaStreamSynchronize can be used as a more fine-grained synchronization mechanism. The program will:
1. Allocate a small array on the host and copy it to the device.
2. Create two CUDA streams.
3. Launch two simple kernels in parallel, each performing a different arithmetic operation on the array.
4. Synchronize each stream individually using cudaStreamSynchronize to ensure the kernels have finished before copying the data back.
5. Compare this approach to using cudaDeviceSynchronize, which would block the entire device, thereby forcing all streams to finish and potentially wasting idle time on other streams or host threads.

Key points to observe:
- cudaDeviceSynchronize blocks all CUDA work, effectively forcing a global barrier.
- cudaStreamSynchronize only blocks the specific stream and does not affect other streams.
- In a multi-stream application, using stream-level synchronization can reduce idle times and improve throughput.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that adds a constant value to each element
__global__ void addConstant(float *data, float constant, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += constant;
    }
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main() {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f; // initialize with 1.0

    // Allocate device memory
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_data, h_in, size, cudaMemcpyHostToDevice));

    // Create two streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Launch first kernel in stream1
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    addConstant<<<gridSize, blockSize, 0, stream1>>>(d_data, 2.0f, N);
    CUDA_CHECK(cudaGetLastError());

    // Launch second kernel in stream2
    addConstant<<<gridSize, blockSize, 0, stream2>>>(d_data, 3.0f, N);
    CUDA_CHECK(cudaGetLastError());

    // Fine‑grained synchronization: wait for each stream individually
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // At this point both kernels have finished, and d_data contains the result.
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_data, size, cudaMemcpyDeviceToHost));

    // Verify results (should be 1 + 2 + 3 = 6 for each element)
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_out[i] - 6.0f) > 1e-5f) {
            success = false;
            printf("Mismatch at index %d: %f\n", i, h_out[i]);
            break;
        }
    }
    if (success) {
        printf("All results correct.\n");
    }

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaFree(d_data));
    free(h_in);
    free(h_out);

    return 0;
}
```