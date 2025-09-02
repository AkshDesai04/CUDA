/*
Aim of the program:
    An alternative to loading 0 is to make the computation loop conditional. Is this more or less efficient?

The thinking:
    In CUDA, loading an accumulator with 0 before a loop is a common pattern. An alternative is to wrap the loop
    inside a conditional that checks whether there is work to do. The question is whether the conditional introduces
    overhead (branch divergence) that outweighs the benefit of not executing a loop that would otherwise iterate 0
    times. This program demonstrates both approaches by summing variable-length segments of an array per thread.
    For each thread, if the segment length is zero the loop would do nothing. We implement two kernels:
        1. kernel_no_cond: always loads acc = 0 and then executes the loop (the loop body is empty if len==0).
        2. kernel_cond: branches on len>0 and only enters the loop if len>0.
    We then time both kernels using CUDA events on a large number of threads and compare the execution times.
    Since all threads in a warp may have different segment lengths, the conditional introduces warp divergence
    that may serialize execution and make the conditional loop less efficient.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Parameters
const int NUM_BLOCKS = 256;
const int THREADS_PER_BLOCK = 256;
const int NUM_THREADS = NUM_BLOCKS * THREADS_PER_BLOCK;
const int MAX_SEG_LEN = 64;   // maximum length of each segment

// Kernel that always loads accumulator = 0 and loops
__global__ void kernel_no_cond(const float* data, const int* seg_len, float* result, int max_seg_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_THREADS) return;

    int len = seg_len[idx];
    int base = idx * max_seg_len;

    float acc = 0.0f;
    for (int i = 0; i < len; ++i) {
        acc += data[base + i];
    }
    result[idx] = acc;
}

// Kernel that branches on seg_len > 0
__global__ void kernel_cond(const float* data, const int* seg_len, float* result, int max_seg_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_THREADS) return;

    int len = seg_len[idx];
    int base = idx * max_seg_len;

    float acc = 0.0f;
    if (len > 0) {
        for (int i = 0; i < len; ++i) {
            acc += data[base + i];
        }
    }
    result[idx] = acc;
}

int main()
{
    // Allocate host memory
    size_t seg_len_h_size = NUM_THREADS * sizeof(int);
    size_t data_h_size = NUM_THREADS * MAX_SEG_LEN * sizeof(float);
    size_t result_h_size = NUM_THREADS * sizeof(float);

    int* seg_len_h = (int*)malloc(seg_len_h_size);
    float* data_h = (float*)malloc(data_h_size);
    float* result_h = (float*)malloc(result_h_size);

    // Initialize data: random lengths between 0 and MAX_SEG_LEN-1
    srand(42);
    for (int i = 0; i < NUM_THREADS; ++i) {
        seg_len_h[i] = rand() % MAX_SEG_LEN; // can be 0
        for (int j = 0; j < MAX_SEG_LEN; ++j) {
            data_h[i * MAX_SEG_LEN + j] = (float)(rand()) / RAND_MAX;
        }
    }

    // Allocate device memory
    int* seg_len_d;
    float* data_d;
    float* result_d;
    CHECK_CUDA(cudaMalloc((void**)&seg_len_d, seg_len_h_size));
    CHECK_CUDA(cudaMalloc((void**)&data_d, data_h_size));
    CHECK_CUDA(cudaMalloc((void**)&result_d, result_h_size));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(seg_len_d, seg_len_h, seg_len_h_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data_d, data_h, data_h_size, cudaMemcpyHostToDevice));

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Run kernel_no_cond
    CHECK_CUDA(cudaEventRecord(start));
    kernel_no_cond<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(data_d, seg_len_d, result_d, MAX_SEG_LEN);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_no_cond;
    CHECK_CUDA(cudaEventElapsedTime(&ms_no_cond, start, stop));

    // Run kernel_cond
    CHECK_CUDA(cudaEventRecord(start));
    kernel_cond<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(data_d, seg_len_d, result_d, MAX_SEG_LEN);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_cond;
    CHECK_CUDA(cudaEventElapsedTime(&ms_cond, start, stop));

    // Copy result back (just to avoid compiler optimizing away)
    CHECK_CUDA(cudaMemcpy(result_h, result_d, result_h_size, cudaMemcpyDeviceToHost));

    // Compute simple checksum to verify correctness
    double sum_no_cond = 0.0;
    double sum_cond = 0.0;
    for (int i = 0; i < NUM_THREADS; ++i) {
        sum_no_cond += result_h[i];
    }
    // Run kernel_cond again to get result for sum_cond
    kernel_cond<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(data_d, seg_len_d, result_d, MAX_SEG_LEN);
    CHECK_CUDA(cudaMemcpy(result_h, result_d, result_h_size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUM_THREADS; ++i) {
        sum_cond += result_h[i];
    }

    printf("Execution time:\n");
    printf("  kernel_no_cond: %.3f ms\n", ms_no_cond);
    printf("  kernel_cond   : %.3f ms\n", ms_cond);
    printf("\nChecksum comparison:\n");
    printf("  sum_no_cond: %.6f\n", sum_no_cond);
    printf("  sum_cond   : %.6f\n", sum_cond);

    // Clean up
    CHECK_CUDA(cudaFree(seg_len_d));
    CHECK_CUDA(cudaFree(data_d));
    CHECK_CUDA(cudaFree(result_d));
    free(seg_len_h);
    free(data_h);
    free(result_h);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
