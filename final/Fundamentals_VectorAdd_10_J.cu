/*
Aim: Explain in a comment why GPUs excel at this type of 'embarrassingly parallel' problem.

Thinking:
The core idea behind the "embarrassingly parallel" class of problems is that each computation can be performed independently, without any need for communication or synchronization with other computations. This property is a perfect match for the GPU architecture for several reasons:

1. **Massive Thread Parallelism**: GPUs contain thousands of lightweight cores (CUDA cores on NVIDIA hardware). When we launch a kernel that spawns a thread for each independent operation, the GPU can execute many of those threads concurrently. This maps one-to-one to the problem's parallel nature.

2. **SIMT Execution Model**: CUDA uses a Single Instruction, Multiple Threads (SIMT) model where warps (groups of 32 threads) execute the same instruction simultaneously. In embarrassingly parallel work, all threads follow identical control flow, ensuring full warp utilization and minimal divergence.

3. **High Memory Bandwidth**: GPUs are designed with a large memory bus and high memory bandwidth. Independent computations that primarily read from and write to global memory can benefit from this bandwidth, especially when each thread accesses distinct memory locations, reducing contention.

4. **Latency Hiding**: GPU hardware can switch between warps rapidly to hide memory access latency. Since each thread works independently, the GPU can keep many warps in flight, ensuring that when one warp stalls waiting for memory, another can use the execution units.

5. **SIMD/SIMT Efficiency**: Because the operations are identical across threads, the GPU can execute them in a tightly synchronized manner, yielding high instruction throughput compared to a CPU that would handle them in a more serialized fashion.

In short, GPUs are engineered to accelerate workloads where a huge number of identical, independent tasks can be performed in parallel. Embarrassingly parallel problems exploit this strength by mapping each independent computation to a GPU thread, allowing the device to keep its many cores busy and achieve massive speedups over serial or lightly parallel CPU implementations. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that performs element-wise addition of two arrays
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel with enough blocks to cover all elements
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) {
            correct = false;
            printf("Mismatch at index %d: %f != %f\n", i, h_C[i], h_A[i] + h_B[i]);
            break;
        }
    }
    printf("Computation %s\n", correct ? "PASSED" : "FAILED");

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
