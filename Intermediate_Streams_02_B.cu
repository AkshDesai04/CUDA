/*
You will need two sets of device buffers, one for each stream.
This comment includes the aim of the program exactly as requested.
Below I explain my thinking process in detail.

1. Purpose of the program
   The goal is to demonstrate the use of two CUDA streams each with its own
   pair of device buffers.  We allocate separate device memory for each
   stream so that operations in one stream do not interfere with the other.
   The program performs a simple vector addition on two distinct input
   vectors in parallel streams.  After the kernels finish we copy the
   results back to the host and verify correctness.

2. Design decisions
   - Host data: Two pairs of input vectors (A1,B1) and (A2,B2).
   - Device data: For stream 1 we allocate dA1,dB1,dC1; for stream 2
     dA2,dB2,dC2.
   - CUDA streams: stream1 and stream2 created with cudaStreamCreate.
   - Kernel: addVectors performs element-wise addition.
   - Execution configuration: 256 threads per block, sufficient blocks to
     cover the vector length.
   - Synchronization: cudaDeviceSynchronize after kernel launches to
     ensure all work is complete before copying back.  In a real
     application we might only synchronize each stream separately,
     but for clarity we wait for both.
   - Error checking: Basic macro to wrap CUDA API calls.

3. Implementation details
   - Use a small vector size (e.g., 1<<20 elements) for demonstration.
   - Initialize host arrays with simple patterns (e.g., i and 2*i).
   - Print a few elements of the output to confirm correctness.
   - Clean up all resources: free device memory, destroy streams, reset
     device.

4. Constraints
   The code is self‑contained in a single .cu file and can be compiled
   with `nvcc`.  No external libraries are required.  The program
   prints the first ten results of each stream to stdout.

The code below follows these design choices and includes comments
explaining each section.  It should compile and run on any CUDA
capable GPU.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA API calls
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// Simple vector addition kernel
__global__ void addVectors(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;   // Size of each vector (1M elements)
    const size_t size = N * sizeof(float);

    // Host memory allocation
    float *h_A1 = (float *)malloc(size);
    float *h_B1 = (float *)malloc(size);
    float *h_C1 = (float *)malloc(size);
    float *h_A2 = (float *)malloc(size);
    float *h_B2 = (float *)malloc(size);
    float *h_C2 = (float *)malloc(size);

    if (!h_A1 || !h_B1 || !h_C1 || !h_A2 || !h_B2 || !h_C2) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_A1[i] = (float)i;
        h_B1[i] = (float)(2 * i);
        h_A2[i] = (float)(3 * i);
        h_B2[i] = (float)(4 * i);
    }

    // Device memory allocation for stream 1
    float *d_A1, *d_B1, *d_C1;
    CUDA_CHECK(cudaMalloc((void **)&d_A1, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B1, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C1, size));

    // Device memory allocation for stream 2
    float *d_A2, *d_B2, *d_C2;
    CUDA_CHECK(cudaMalloc((void **)&d_A2, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B2, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C2, size));

    // Create streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Asynchronously copy data to device for stream 1
    CUDA_CHECK(cudaMemcpyAsync(d_A1, h_A1, size, cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_B1, h_B1, size, cudaMemcpyHostToDevice, stream1));

    // Asynchronously copy data to device for stream 2
    CUDA_CHECK(cudaMemcpyAsync(d_A2, h_A2, size, cudaMemcpyHostToDevice, stream2));
    CUDA_CHECK(cudaMemcpyAsync(d_B2, h_B2, size, cudaMemcpyHostToDevice, stream2));

    // Launch kernels on each stream
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    addVectors<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A1, d_B1, d_C1, N);
    addVectors<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_A2, d_B2, d_C2, N);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpyAsync(h_C1, d_C1, size, cudaMemcpyDeviceToHost, stream1));
    CUDA_CHECK(cudaMemcpyAsync(h_C2, d_C2, size, cudaMemcpyDeviceToHost, stream2));

    // Wait for all operations to finish
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // Verify results (print first 10 elements from each)
    printf("Stream 1 results (first 10 elements):\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_C1[%d] = %f\n", i, h_C1[i]);
    }

    printf("\nStream 2 results (first 10 elements):\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_C2[%d] = %f\n", i, h_C2[i]);
    }

    // Clean up
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    CUDA_CHECK(cudaFree(d_A1));
    CUDA_CHECK(cudaFree(d_B1));
    CUDA_CHECK(cudaFree(d_C1));
    CUDA_CHECK(cudaFree(d_A2));
    CUDA_CHECK(cudaFree(d_B2));
    CUDA_CHECK(cudaFree(d_C2));

    free(h_A1);
    free(h_B1);
    free(h_C1);
    free(h_A2);
    free(h_B2);
    free(h_C2);

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
