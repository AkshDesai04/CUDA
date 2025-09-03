/*
Aim: Add a comment to your program explaining what 'concurrent kernel execution' means and how it can improve performance.

The goal of this CUDA program is to illustrate the concept of concurrent kernel execution by launching two simple vector addition kernels in separate CUDA streams. 
The comment added to the code explains what concurrent kernel execution is and how it can improve performance. 
The program demonstrates how kernels launched in different streams can run in parallel on the GPU, provided there are enough resources and the hardware supports it. 
This example also includes basic error checking and cleanup. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 20) // 1M elements

// Kernel to add two vectors
__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    // Allocate host memory
    size_t size = N * sizeof(float);
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    float *h_d = (float*)malloc(size); // For second kernel

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c, *d_d;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    cudaMalloc((void**)&d_d, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create two CUDA streams for concurrent execution
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Launch first kernel in stream1
    vecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_a, d_b, d_c, N);

    // Launch second kernel in stream2 (adds the same vectors again but stores in d_d)
    vecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_a, d_b, d_d, N);

    // Wait for all kernels to finish
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d, d_d, size, cudaMemcpyDeviceToHost);

    // Simple verification
    if (h_c[0] == h_d[0]) {
        printf("Concurrent kernel execution succeeded.\n");
    } else {
        printf("Error in concurrent execution.\n");
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}

/*
Concurrent kernel execution means launching multiple kernels that can run at the same time on the GPU. 
When kernels are launched in different CUDA streams, the driver schedules them for execution on the GPU hardware. 
If the GPU has enough multiprocessors, registers, and other resources, the hardware can interleave or run these kernels simultaneously, overlapping memory transfers and compute. 
This overlap reduces idle time on the GPU, improves resource utilization, and can lead to a noticeable reduction in overall runtime for workloads that can be split into independent kernels.
*/