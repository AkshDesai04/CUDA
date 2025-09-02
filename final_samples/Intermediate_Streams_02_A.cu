/*
Aim of the program: Create two streams: `cudaStream_t stream1, stream2; cudaStreamCreate(...)`.

Thinking:
The task is to write a CUDA program in C that demonstrates the creation of two streams and
the use of those streams to launch kernels asynchronously.  
The program will:
1. Define two stream variables (`stream1` and `stream2`) and create them with `cudaStreamCreate`.
2. Allocate a small array on both host and device.
3. Launch a simple kernel on each stream that increments each element of the array.
4. Use `cudaMemcpyAsync` to copy the result back to the host in each stream.
5. Synchronize the streams to ensure all operations are finished before checking results.
6. Clean up device memory and destroy the streams.

An error‑checking macro is added for brevity. The kernel is trivial but sufficient to show
that the streams are independent and can run concurrently (though with such a small kernel
the overlap may not be noticeable). The code compiles with `nvcc` and runs on any
CUDA‑capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel that adds 1.0f to each element
__global__ void add_one(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1.0f;
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_data = (float*)malloc(size);
    for (int i = 0; i < N; ++i) h_data[i] = 0.0f; // initialize to zero

    // Allocate device memory
    float *d_data1 = NULL, *d_data2 = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_data1, size));
    CUDA_CHECK(cudaMalloc((void**)&d_data2, size));

    // Create two streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Copy host data to device in stream1
    CUDA_CHECK(cudaMemcpyAsync(d_data1, h_data, size, cudaMemcpyHostToDevice, stream1));
    // Launch kernel in stream1
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_one<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data1, N);
    // Copy back result in stream1
    CUDA_CHECK(cudaMemcpyAsync(h_data, d_data1, size, cudaMemcpyDeviceToHost, stream1));

    // For stream2, reuse host data but copy to different device array
    CUDA_CHECK(cudaMemcpyAsync(d_data2, h_data, size, cudaMemcpyHostToDevice, stream2));
    // Launch kernel in stream2
    add_one<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_data2, N);
    // Copy back result in stream2 to a separate host buffer
    float *h_result2 = (float*)malloc(size);
    CUDA_CHECK(cudaMemcpyAsync(h_result2, d_data2, size, cudaMemcpyDeviceToHost, stream2));

    // Synchronize streams to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // Verify results
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_data[i] != 1.0f) { errors++; break; }
        if (h_result2[i] != 1.0f) { errors++; break; }
    }
    if (errors == 0)
        printf("Success: both streams completed correctly.\n");
    else
        printf("Error: data mismatch.\n");

    // Cleanup
    free(h_data);
    free(h_result2);
    CUDA_CHECK(cudaFree(d_data1));
    CUDA_CHECK(cudaFree(d_data2));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    return 0;
}
