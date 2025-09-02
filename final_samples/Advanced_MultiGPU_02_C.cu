```cuda
/* 
Aim of the program: 
"Generalize the program to split the work across all available GPUs on the system."

Thinking and design notes:
1. The goal is to demonstrate how to discover the number of GPUs in a system using
   `cudaGetDeviceCount`, partition a large workload (vector addition) across those
   devices, and then gather the results back on the host.
2. The workload chosen is simple: element-wise addition of two large floating-point arrays.
3. Each GPU receives a contiguous block of the arrays. If the array size is not
   perfectly divisible by the number of GPUs, the last GPU receives the remainder.
4. For each GPU we:
   a. Set the device with `cudaSetDevice`.
   b. Allocate device memory for its chunk of each array.
   c. Copy the chunk from the host to the device (using `cudaMemcpyAsync` for overlap).
   d. Launch a simple kernel that adds the two vectors.
   e. Copy the result back to a host buffer (again async).
5. We use a separate CUDA stream per device to allow concurrent operations where
   the hardware permits.
6. After launching kernels on all devices, we wait for all streams to finish with
   `cudaDeviceSynchronize`. Then we assemble the final result from the per-GPU
   chunks and optionally verify it against a CPU reference.
7. Error handling is minimal but uses a macro to check CUDA calls and abort on
   failure, keeping the example readable.
8. The program is self-contained, can be compiled with `nvcc` and will run on any
   system with multiple GPUs (or fall back to a single GPU if only one is present).
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <iostream>

#define CHECK_CUDA(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

// Simple kernel for element-wise vector addition
__global__ void vecAdd(const float* A, const float* B, float* C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

int main() {
    const size_t N = 1 << 28;  // Example size (~268 million elements, ~1GB each)
    std::cout << "Vector size: " << N << " elements (" << (N * sizeof(float) / (1024*1024*1024)) << " GB)\n";

    // Allocate host memory
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Determine number of GPUs
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }
    std::cout << "Detected " << deviceCount << " CUDA device(s).\n";

    // Compute chunk sizes for each GPU
    std::vector<size_t> chunkStarts(deviceCount);
    std::vector<size_t> chunkSizes(deviceCount);
    size_t baseChunk = N / deviceCount;
    size_t remainder = N % deviceCount;
    size_t offset = 0;
    for (int d = 0; d < deviceCount; ++d) {
        size_t chunk = baseChunk + (d < remainder ? 1 : 0);
        chunkStarts[d] = offset;
        chunkSizes[d] = chunk;
        offset += chunk;
    }

    // Create a vector of streams, one per device
    std::vector<cudaStream_t> streams(deviceCount);
    for (int d = 0; d < deviceCount; ++d) {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaStreamCreate(&streams[d]));
    }

    // For each GPU: allocate device memory, copy data, launch kernel, copy back
    for (int d = 0; d < deviceCount; ++d) {
        CHECK_CUDA(cudaSetDevice(d));
        size_t chunkSize = chunkSizes[d];
        size_t chunkStart = chunkStarts[d];

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, chunkSize * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, chunkSize * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, chunkSize * sizeof(float)));

        // Copy input chunks to device asynchronously
        CHECK_CUDA(cudaMemcpyAsync(d_A, h_A + chunkStart, chunkSize * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[d]));
        CHECK_CUDA(cudaMemcpyAsync(d_B, h_B + chunkStart, chunkSize * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[d]));

        // Launch kernel
        size_t threadsPerBlock = 256;
        size_t blocksPerGrid = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;
        vecAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[d]>>>(d_A, d_B, d_C, chunkSize);
        CHECK_CUDA(cudaGetLastError());

        // Copy result back asynchronously
        CHECK_CUDA(cudaMemcpyAsync(h_C + chunkStart, d_C, chunkSize * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[d]));

        // Free device memory
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }

    // Synchronize all streams (wait for all GPUs to finish)
    for (int d = 0; d < deviceCount; ++d) {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaStreamSynchronize(streams[d]));
        CHECK_CUDA(cudaStreamDestroy(streams[d]));
    }

    // Simple verification on host
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = false;
            fprintf(stderr, "Mismatch at index %zu: %f != %f + %f\n",
                    i, h_C[i], h_A[i], h_B[i]);
            break;
        }
    }
    if (correct) {
        std::cout << "Verification succeeded: all results are correct.\n";
    } else {
        std::cout << "Verification failed.\n";
    }

    // Clean up host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```