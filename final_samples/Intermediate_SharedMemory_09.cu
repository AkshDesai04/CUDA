```cuda
/* 
 * Aim of the program (word-for-word as requested):
 *   "Analyze shared memory bank conflicts. Write a kernel where threads access shared memory with a stride of 1 (shared_mem[threadIdx.x]) and another where they access with a stride equal to the number of banks (e.g., 32, shared_mem[32 * threadIdx.x]). Profile both and explain the performance difference."
 *
 * My thinking:
 *   - Shared memory on modern GPUs (e.g., sm_60+) is divided into 32 banks. Each bank can serve one 32-bit word per cycle.
 *   - If multiple threads in the same warp access the same bank, a conflict occurs and the accesses are serialized.
 *   - In the stride-1 kernel, each thread accesses a consecutive address: shared_mem[threadIdx.x]. Within a warp of 32 threads, these addresses map to banks 0–31 respectively, so there are no conflicts.
 *   - In the stride-32 kernel, each thread accesses shared_mem[32 * threadIdx.x]. Within a warp of 32 threads, all these addresses map to bank 0, causing a full 32-way conflict and serialization.
 *   - To measure the impact, I use a loop inside each kernel that performs many shared memory accesses, so the kernel execution time is significant enough to be measured accurately with cudaEvent.
 *   - I allocate a static shared memory array of size 4096, which is sufficient for both kernels (stride-1 uses first 128 entries; stride-32 uses up to index 32*(128-1)=4032).
 *   - I run each kernel once with a large number of iterations (100,000) to ensure the kernel takes enough time to be measured reliably.
 *   - Finally, I output the elapsed times and discuss the observed performance difference.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 128          // number of threads per block
#define NUM_ITER 100000        // iterations inside kernel for measurable time
#define NUM_RUNS 1             // how many times to launch each kernel (1 is enough due to many iterations)

// Kernel where each thread accesses shared memory with stride 1
__global__ void kernel_stride1(int *out, int iterations)
{
    int tid = threadIdx.x;
    __shared__ int smem[4096];          // static shared memory array

    // Write once
    smem[tid] = tid;
    __syncthreads();

    // Perform many accesses to shared memory
    int sum = 0;
    for (int i = 0; i < iterations; ++i) {
        sum += smem[tid];
    }

    out[tid] = sum;
}

// Kernel where each thread accesses shared memory with stride equal to number of banks (32)
__global__ void kernel_stride32(int *out, int iterations)
{
    int tid = threadIdx.x;
    __shared__ int smem[4096];

    // Write once with stride 32
    smem[32 * tid] = tid;
    __syncthreads();

    // Perform many accesses to shared memory
    int sum = 0;
    for (int i = 0; i < iterations; ++i) {
        sum += smem[32 * tid];
    }

    out[tid] = sum;
}

// Helper to check CUDA errors
inline void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Allocate device output arrays
    int *d_out1, *d_out2;
    size_t outSize = BLOCK_DIM * sizeof(int);
    checkCuda(cudaMalloc(&d_out1, outSize), "cudaMalloc d_out1");
    checkCuda(cudaMalloc(&d_out2, outSize), "cudaMalloc d_out2");

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop),  "cudaEventCreate stop");

    float elapsed1 = 0.0f;
    float elapsed2 = 0.0f;

    // ----------------------------
    // Run kernel with stride 1
    // ----------------------------
    for (int run = 0; run < NUM_RUNS; ++run) {
        checkCuda(cudaEventRecord(start, 0), "cudaEventRecord start stride1");
        kernel_stride1<<<1, BLOCK_DIM>>>(d_out1, NUM_ITER);
        checkCuda(cudaEventRecord(stop, 0),  "cudaEventRecord stop stride1");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stride1");
        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime stride1");
        elapsed1 += ms;
    }

    // ----------------------------
    // Run kernel with stride 32
    // ----------------------------
    for (int run = 0; run < NUM_RUNS; ++run) {
        checkCuda(cudaEventRecord(start, 0), "cudaEventRecord start stride32");
        kernel_stride32<<<1, BLOCK_DIM>>>(d_out2, NUM_ITER);
        checkCuda(cudaEventRecord(stop, 0),  "cudaEventRecord stop stride32");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stride32");
        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime stride32");
        elapsed2 += ms;
    }

    // Print timing results
    printf("Average elapsed time over %d run(s):\n", NUM_RUNS);
    printf("  Stride 1 kernel : %f ms\n", elapsed1 / NUM_RUNS);
    printf("  Stride 32 kernel: %f ms\n", elapsed2 / NUM_RUNS);

    // Optional: copy results back and verify (not required for performance analysis)
    int h_out1[BLOCK_DIM];
    int h_out2[BLOCK_DIM];
    checkCuda(cudaMemcpy(h_out1, d_out1, outSize, cudaMemcpyDeviceToHost), "cudaMemcpy d_out1");
    checkCuda(cudaMemcpy(h_out2, d_out2, outSize, cudaMemcpyDeviceToHost), "cudaMemcpy d_out2");

    // Clean up
    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```