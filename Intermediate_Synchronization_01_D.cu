```c
/*
Aim of the program:
An alternative to a second kernel is to copy the d_partial_sums array back to the host and sum it on the CPU.

Thinking:
The goal is to illustrate how a typical two-stage reduction (first kernel computes block-wise partial sums, second kernel sums those partial sums) can be replaced by copying the block-wise results back to host and performing the final summation on the CPU. 
The program will:

1. Create a large input array on the host and transfer it to the device.
2. Launch a reduction kernel that computes partial sums for each block and stores one sum per block in d_partial_sums.
3. Copy d_partial_sums back to the host.
4. Perform a final summation on the CPU.
5. Verify the result against a straightforward CPU sum.

The reduction kernel uses shared memory to perform a warp-synchronous reduction within each block. The block size is chosen as 256 threads. 
We include error checking macros to simplify debugging. The code is self-contained and can be compiled with nvcc as follows:
    nvcc -o reduction_sum reduction_sum.cu
Then run with:
    ./reduction_sum
It will print the computed sum, the expected sum, and a success/failure message.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel: per-block reduction to compute partial sums
__global__ void reduce_sum_kernel(const float *input, float *partial_sums, size_t n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x; // load two elements per thread

    float sum = 0.0f;

    // Load two elements into shared memory (if within bounds)
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Unroll the last warp
    if (tid < 32) {
        volatile float *vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Write result for this block to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

int main(void) {
    // Parameters
    const size_t N = 1 << 24; // 16M elements
    const size_t bytes = N * sizeof(float);
    const unsigned int threadsPerBlock = 256;
    const unsigned int elementsPerThread = 2;
    const unsigned int blocksPerGrid = (N + threadsPerBlock * elementsPerThread - 1) / (threadsPerBlock * elementsPerThread);

    // Host allocation
    float *h_input = (float*)malloc(bytes);
    float *h_partial_sums = (float*)malloc(blocksPerGrid * sizeof(float));

    // Initialize input array
    for (size_t i = 0; i < N; ++i) {
        h_input[i] = 1.0f; // Simple value for easy verification
    }

    // Device allocation
    float *d_input = NULL;
    float *d_partial_sums = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_partial_sums, blocksPerGrid * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    reduce_sum_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_partial_sums, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums back to host
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    // Final summation on CPU
    float cpu_sum = 0.0f;
    for (unsigned int i = 0; i < blocksPerGrid; ++i) {
        cpu_sum += h_partial_sums[i];
    }

    // Verify against expected sum
    float expected_sum = N * 1.0f; // Since we initialized all to 1.0f
    printf("CPU sum: %f\n", cpu_sum);
    printf("Expected sum: %f\n", expected_sum);
    printf("Result %s\n", (abs(cpu_sum - expected_sum) < 1e-5) ? "PASS" : "FAIL");

    // Clean up
    free(h_input);
    free(h_partial_sums);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial_sums));

    return 0;
}
```