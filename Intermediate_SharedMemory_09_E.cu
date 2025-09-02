```cuda
/*
Aim of the program: 
Write a kernel where `output[i] = shared_mem[threadIdx.x * 32]`. This should cause severe 32-way bank conflicts. Profile it and compare to the stride-1 case.

Thinking:
1. We need two kernels: one that forces 32‑way bank conflicts by accessing shared memory at an offset of 32 times the thread index, and one that accesses sequential elements to avoid conflicts.
2. Shared memory is allocated dynamically so that the conflict kernel can have enough slots (BLOCK_SIZE * 32). The no‑conflict kernel only needs BLOCK_SIZE slots.
3. Each thread will first load its input element into shared memory, then synchronize. The conflict kernel then reads from `shared_mem[threadIdx.x * 32]` while the no‑conflict kernel reads from `shared_mem[threadIdx.x]`.
4. We will time both kernels using CUDA events, run each a few times for stability, and print the average execution times.
5. A simple correctness check compares the output arrays from both kernels; they should be identical if the data transfer to shared memory is correct.
6. The program uses standard CUDA C features, error checking, and is self‑contained in a single .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CHECK_CUDA(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

__global__ void kernel_conflict(float *output, const float *input, int N)
{
    extern __shared__ float s_mem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Load into shared memory
        s_mem[threadIdx.x] = input[idx];
        __syncthreads();

        // Force 32-way bank conflicts
        float val = s_mem[threadIdx.x * 32];
        output[idx] = val;
    }
}

__global__ void kernel_no_conflict(float *output, const float *input, int N)
{
    extern __shared__ float s_mem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Load into shared memory
        s_mem[threadIdx.x] = input[idx];
        __syncthreads();

        // No bank conflicts
        float val = s_mem[threadIdx.x];
        output[idx] = val;
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output_conflict = (float*)malloc(N * sizeof(float));
    float *h_output_noconflict = (float*)malloc(N * sizeof(float));

    // Initialize input
    for (int i = 0; i < N; ++i) h_input[i] = (float)i;

    // Allocate device memory
    float *d_input, *d_output_conflict, *d_output_noconflict;
    CHECK_CUDA(cudaMalloc((void**)&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output_conflict, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output_noconflict, N * sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Number of repeats for stable timing
    const int repeats = 10;

    // Timing conflict kernel
    float conflict_time = 0.0f;
    for (int r = 0; r < repeats; ++r) {
        CHECK_CUDA(cudaEventRecord(start));
        kernel_conflict<<<blocks, threadsPerBlock, threadsPerBlock * 32 * sizeof(float)>>>(
            d_output_conflict, d_input, N);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        conflict_time += ms;
    }
    conflict_time /= repeats;

    // Timing no-conflict kernel
    float noconflict_time = 0.0f;
    for (int r = 0; r < repeats; ++r) {
        CHECK_CUDA(cudaEventRecord(start));
        kernel_no_conflict<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            d_output_noconflict, d_input, N);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        noconflict_time += ms;
    }
    noconflict_time /= repeats;

    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_output_conflict, d_output_conflict, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_noconflict, d_output_noconflict, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify correctness
    int mismatch = 0;
    for (int i = 0; i < N; ++i) {
        if (h_output_conflict[i] != h_output_noconflict[i]) {
            mismatch = 1;
            if (i < 10) // print first few mismatches
                printf("Mismatch at %d: conflict=%f, noconflict=%f\n",
                       i, h_output_conflict[i], h_output_noconflict[i]);
        }
    }
    if (mismatch) printf("Outputs differ!\n");
    else printf("Outputs match.\n");

    // Print timing results
    printf("Average execution time over %d runs:\n", repeats);
    printf("  Conflict kernel      : %f ms\n", conflict_time);
    printf("  No-conflict kernel   : %f ms\n", noconflict_time);
    printf("Speedup (noconflict / conflict) : %f\n",
           conflict_time / noconflict_time);

    // Clean up
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_conflict));
    CHECK_CUDA(cudaFree(d_output_noconflict));
    free(h_input);
    free(h_output_conflict);
    free(h_output_noconflict);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```