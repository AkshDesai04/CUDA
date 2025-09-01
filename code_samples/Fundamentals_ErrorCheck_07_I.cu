/*
Why does CUDA use an asynchronous model? (Performance, allows CPU and GPU to work in parallel).

My thinking: CUDA's asynchronous execution model is designed to overlap CPU and GPU work to maximize throughput and hide latency. By issuing commands (memory copies, kernel launches) that return immediately, the CPU can perform other tasks (e.g., preparing data, handling I/O) while the GPU executes its workload. The underlying hardware scheduler dispatches these commands onto hardware units, and streams can be used to control ordering and concurrency. This model is essential for scaling applications to high performance and making full use of the GPU's compute capabilities.

Implementation: This example demonstrates asynchronous memory copy and kernel execution using CUDA streams. The host program launches a kernel to add two vectors, performing memory copies and kernel execution asynchronously, and then synchronizes at the end. Timing measurements with cudaEvent_t illustrate overlapping of data transfer and computation, reflecting the asynchronous model.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Simple vector addition kernel
__global__ void vectorAdd(const float *a, const float *b, float *c, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

int main(void) {
    const size_t N = 1 << 24;           // 16M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_c, bytes));

    // Create a stream for asynchronous operations
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Create CUDA events for timing
    cudaEvent_t start, stop, copyIn, copyOut;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&copyIn));
    CHECK_CUDA(cudaEventCreate(&copyOut));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, stream));

    // Asynchronously copy input arrays to device
    CHECK_CUDA(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaEventRecord(copyIn, stream));

    // Launch kernel asynchronously in the same stream
    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());

    // Asynchronously copy result back to host
    CHECK_CUDA(cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaEventRecord(copyOut, stream));

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, stream));

    // Wait for all operations in the stream to complete
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Compute elapsed times
    float timeTotal, timeCopyIn, timeCopyOut;
    CHECK_CUDA(cudaEventElapsedTime(&timeTotal, start, stop));
    CHECK_CUDA(cudaEventElapsedTime(&timeCopyIn, start, copyIn));
    CHECK_CUDA(cudaEventElapsedTime(&timeCopyOut, copyOut, stop));

    printf("Total elapsed time: %f ms\n", timeTotal);
    printf("Time for host->device copies: %f ms\n", timeCopyIn);
    printf("Time for device->host copy: %f ms\n", timeCopyOut);

    // Verify results
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
            correct = false;
            printf("Mismatch at index %zu: %f != %f\n", i, h_c[i], h_a[i] + h_b[i]);
            break;
        }
    }
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(copyIn));
    CHECK_CUDA(cudaEventDestroy(copyOut));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
