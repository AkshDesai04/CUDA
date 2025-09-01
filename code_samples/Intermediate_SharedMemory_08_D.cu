```cuda
/*
This reduction pattern has bank conflicts in the first few iterations. A conflict-free version has the stride s decrease from blockDim.x / 2 but the access is s_data[i] += s_data[i+s]. Compare the performance.

To evaluate the impact of shared memory bank conflicts on a classic tree‑based reduction we implement two CUDA kernels:
1. `reduce_conflict` – a straightforward reduction that uses `s_data[i] += s_data[i + stride]`.  
   In the early iterations many threads access the same shared‑memory bank because the stride is a power‑of‑two, causing serialization.
2. `reduce_conflict_free` – a bank‑conflict‑free variant. We pad the shared array by one element, offset all indices by +1, and keep the same stride pattern.  
   Because no two active threads address the same bank during the first half of the reduction, the kernel runs faster.

Both kernels reduce a single block of 1024 elements and output the sum to global memory.  
We time each kernel over many iterations (1000) to average out noise, then compare the elapsed times.  
The CPU sum is also computed to verify correctness.

The program demonstrates how a simple change in shared‑memory indexing eliminates bank conflicts and improves performance. The timings printed by the program illustrate this gain.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 1024
#define ITER 1000
#define N BLOCK_SIZE  // Number of elements per block

// Error checking macro
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",            \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                                \
    } while (0)

// Kernel with potential bank conflicts
__global__ void reduce_conflict(const float *input, float *output, int N) {
    extern __shared__ float s_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load input into shared memory
    s_data[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) output[blockIdx.x] = s_data[0];
}

// Conflict‑free reduction (padding by one element)
__global__ void reduce_conflict_free(const float *input, float *output, int N) {
    extern __shared__ float s_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Pad shared memory by one to shift indices
    s_data[tid + 1] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Tree reduction with offset
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid + 1] += s_data[tid + stride + 1];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) output[blockIdx.x] = s_data[1];
}

int main(void) {
    // Host allocation
    float *h_input = (float*)malloc(N * sizeof(float));
    if (!h_input) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input with random values
    srand(42);
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)rand() / RAND_MAX;
    }

    // Compute reference sum on CPU
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; ++i) cpu_sum += h_input[i];
    printf("CPU sum: %f\n", cpu_sum);

    // Device allocation
    float *d_input, *d_output_conflict, *d_output_no_conflict;
    CHECK_CUDA(cudaMalloc((void**)&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output_conflict, sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output_no_conflict, sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ----------------- Timing reduce_conflict -----------------
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < ITER; ++i) {
        reduce_conflict<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            d_input, d_output_conflict, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_conflict;
    CHECK_CUDA(cudaEventElapsedTime(&ms_conflict, start, stop));
    printf("Kernel 'reduce_conflict' average time: %f ms\n", ms_conflict / ITER);

    // ----------------- Timing reduce_conflict_free -----------------
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < ITER; ++i) {
        reduce_conflict_free<<<1, BLOCK_SIZE, (BLOCK_SIZE + 1) * sizeof(float)>>>(
            d_input, d_output_no_conflict, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_no_conflict;
    CHECK_CUDA(cudaEventElapsedTime(&ms_no_conflict, start, stop));
    printf("Kernel 'reduce_conflict_free' average time: %f ms\n", ms_no_conflict / ITER);

    // Retrieve results
    float host_sum_conflict, host_sum_no_conflict;
    CHECK_CUDA(cudaMemcpy(&host_sum_conflict, d_output_conflict, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&host_sum_no_conflict, d_output_no_conflict, sizeof(float), cudaMemcpyDeviceToHost));

    printf("GPU sum (conflict): %f\n", host_sum_conflict);
    printf("GPU sum (conflict‑free): %f\n", host_sum_no_conflict);

    // Verify correctness
    const float eps = 1e-5f;
    if (abs(host_sum_conflict - cpu_sum) > eps) {
        fprintf(stderr, "Mismatch in conflict kernel result!\n");
    }
    if (abs(host_sum_no_conflict - cpu_sum) > eps) {
        fprintf(stderr, "Mismatch in conflict‑free kernel result!\n");
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_conflict));
    CHECK_CUDA(cudaFree(d_output_no_conflict));
    free(h_input);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```