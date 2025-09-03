/*
Aim of the program: Use `cudaMemcpyAsync` and pass the stream argument to all copies and kernel launches.

Thinking:
- The requirement is to demonstrate the use of `cudaMemcpyAsync` with stream synchronization. 
- We will create a simple vector addition example (C = A + B) using CUDA streams.
- Allocate host memory for arrays A, B, and C. Fill A and B with known values.
- Allocate device memory for the same arrays.
- Create a single CUDA stream (`cudaStream_t stream`) and use it for all memory copies and kernel launches.
- Use `cudaMemcpyAsync` for host-to-device and device-to-host copies, passing the stream as the last argument.
- Launch the vector addition kernel asynchronously with the same stream.
- After launching the kernel, copy the result back to host asynchronously, again using the stream.
- Synchronize the stream with `cudaStreamSynchronize` to ensure all operations complete before we access the results.
- Print a few elements of the result to verify correctness.
- Perform error checking after each CUDA API call and kernel launch.
- Clean up device memory, host memory, and destroy the stream before exiting.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void vecAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main(void) {
    const int N = 1 << 20;            // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    // Create a CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Asynchronously copy data from host to device
    CHECK_CUDA(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, stream));

    // Kernel launch parameters
    int threads = 256;
    int blocks   = (N + threads - 1) / threads;

    // Launch kernel asynchronously in the stream
    vecAdd<<<blocks, threads, 0, stream>>>(d_A, d_B, d_C, N);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed\n");
        return EXIT_FAILURE;
    }

    // Asynchronously copy result back to host
    CHECK_CUDA(cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, stream));

    // Wait for all operations in the stream to finish
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Verify a few results
    printf("Verification:\n");
    for (int i = 0; i < 5; ++i) {
        printf("C[%d] = %f (expected %f)\n", i, h_C[i], h_A[i] + h_B[i]);
    }

    // Clean up
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
