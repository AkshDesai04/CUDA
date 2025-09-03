```cuda
/*
Aim of the program:
Use `cudaMemAdvise` to give the driver hints about data usage patterns (e.g., set a vector to `cudaMemAdviseSetReadMostly` on the device).

Thinking:
The goal is to demonstrate how to use `cudaMemAdvise` to inform the CUDA driver about how memory will be used. In this example we allocate a large vector on the device, inform the driver that it will be read mostly by the GPU, and then launch a simple kernel that reads the vector. We also illustrate the usage of `cudaMemAdviseSetPreferredLocation` to hint that the memory resides on the GPU. Proper error checking is performed after each CUDA API call to ensure correctness. The kernel simply sums the elements of the vector and stores the result in a separate output array. After kernel execution we copy the result back to the host and print it. The code is self‑contained and can be compiled with `nvcc example.cu -o example`. */

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Simple kernel that sums elements of an array
__global__ void sum_kernel(const float *input, float *output, size_t N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < N) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to output
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main(void) {
    const size_t N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    float *d_vec = NULL;
    float *d_partial_sums = NULL;

    // Allocate device memory for input vector
    CHECK_CUDA(cudaMalloc((void **)&d_vec, bytes));
    // Allocate device memory for partial sums (one per block)
    const size_t numBlocks = 256;
    CHECK_CUDA(cudaMalloc((void **)&d_partial_sums, numBlocks * sizeof(float)));

    // Fill the input vector on host and copy to device
    float *h_vec = (float *)malloc(bytes);
    for (size_t i = 0; i < N; ++i) {
        h_vec[i] = 1.0f; // simple constant value for demo
    }
    CHECK_CUDA(cudaMemcpy(d_vec, h_vec, bytes, cudaMemcpyHostToDevice));
    free(h_vec);

    // Provide usage hints to the driver
    // Hint that the memory will be read mostly by GPU kernels
    CHECK_CUDA(cudaMemAdvise(d_vec, bytes, cudaMemAdviseSetReadMostly, 0));
    // Hint that the memory prefers to stay on the GPU (device 0)
    CHECK_CUDA(cudaMemAdvise(d_vec, bytes, cudaMemAdviseSetPreferredLocation, 0));

    // Launch kernel to compute partial sums
    const size_t threadsPerBlock = 256;
    size_t sharedMemBytes = threadsPerBlock * sizeof(float);
    sum_kernel<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(d_vec, d_partial_sums, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy partial sums back to host and compute final sum
    float *h_partial_sums = (float *)malloc(numBlocks * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_partial_sums, d_partial_sums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    float total_sum = 0.0f;
    for (size_t i = 0; i < numBlocks; ++i) {
        total_sum += h_partial_sums[i];
    }

    printf("Total sum of vector elements: %f\n", total_sum);

    // Clean up
    free(h_partial_sums);
    CHECK_CUDA(cudaFree(d_vec));
    CHECK_CUDA(cudaFree(d_partial_sums));

    return 0;
}
```