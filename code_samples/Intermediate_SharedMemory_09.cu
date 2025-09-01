```cuda
/*
 * Aim: Analyze shared memory bank conflicts. Write a kernel where threads access shared memory with a stride of 1 (`shared_mem[threadIdx.x]`) and another where they access with a stride equal to the number of banks (e.g., 32, `shared_mem[32 * threadIdx.x]`). Profile both and explain the performance difference.
 *
 * Thinking:
 *
 * 1. Modern NVIDIA GPUs divide shared memory into 32 banks, each 4 bytes wide.
 *    The bank index for a word-aligned access is `bank = (address / 4) % 32`.
 * 2. When each thread in a warp accesses consecutive words (`shared_mem[threadIdx.x]`),
 *    the bank indices cycle through 0–31, so every thread touches a distinct bank.
 *    This is the ideal case – no bank conflicts and full warp throughput.
 * 3. If the stride equals the number of banks (`shared_mem[32 * threadIdx.x]`),
 *    the address for every thread in a warp falls on the same bank (`bank = 0`),
 *    leading to a 32‑way conflict. The warp serializes those accesses,
 *    dramatically reducing performance.
 * 4. To demonstrate this, we launch two kernels with identical logic
 *    except for the stride used to index shared memory.
 *    Each kernel copies a global array to shared memory, synchronizes,
 *    and writes the data back to a second global array.
 * 5. We use CUDA events to time the kernels and compare execution times.
 * 6. After running, we expect the stride‑1 kernel to finish noticeably faster
 *    than the stride‑32 kernel, confirming the theoretical bank conflict penalty.
 *
 * Implementation details:
 * - BLOCK_SIZE = 128 threads per block.
 * - For the stride‑32 kernel we allocate a larger shared array
 *   (`BLOCK_SIZE * 32` floats) to provide distinct indices.
 * - Input data is filled with arbitrary values; output is checked for equality.
 * - Timing uses cudaEventRecord and cudaEventElapsedTime.
 *
 * Build and run:
 *   nvcc -arch=sm_75 -o bank_conflict_analysis bank_conflict_analysis.cu
 *   ./bank_conflict_analysis
 */

#include <cstdio>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128
#define NUM_BANKS 32
#define N  (1 << 20)   // 1M elements

// Error checking macro
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Kernel with stride 1 (no bank conflict)
__global__ void kernel_stride1(const float *g_in, float *g_out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Shared memory with one element per thread
    __shared__ float s_mem[BLOCK_SIZE];

    float val = g_in[idx];
    s_mem[threadIdx.x] = val;           // stride 1 access
    __syncthreads();

    g_out[idx] = s_mem[threadIdx.x];
}

// Kernel with stride equal to number of banks (32)
// All threads in a warp access the same bank -> conflict
__global__ void kernel_stride32(const float *g_in, float *g_out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Allocate larger shared memory to accommodate stride 32
    __shared__ float s_mem[BLOCK_SIZE * NUM_BANKS];

    float val = g_in[idx];
    s_mem[NUM_BANKS * threadIdx.x] = val;   // stride 32 access
    __syncthreads();

    g_out[idx] = s_mem[NUM_BANKS * threadIdx.x];
}

int main(void)
{
    // Allocate host memory
    size_t size_bytes = N * sizeof(float);
    float *h_in  = (float*)malloc(size_bytes);
    float *h_out1 = (float*)malloc(size_bytes);
    float *h_out2 = (float*)malloc(size_bytes);

    // Initialize input
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

    // Allocate device memory
    float *d_in, *d_out1, *d_out2;
    CUDA_CHECK(cudaMalloc(&d_in,  size_bytes));
    CUDA_CHECK(cudaMalloc(&d_out1, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_out2, size_bytes));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size_bytes, cudaMemcpyHostToDevice));

    // Launch parameters
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ------------------ Kernel with stride 1 ------------------
    CUDA_CHECK(cudaEventRecord(start));
    kernel_stride1<<<grid, block>>>(d_in, d_out1, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_stride1 = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_stride1, start, stop));
    printf("Kernel stride 1 time: %f ms\n", ms_stride1);

    // ------------------ Kernel with stride 32 ------------------
    CUDA_CHECK(cudaEventRecord(start));
    kernel_stride32<<<grid, block>>>(d_in, d_out2, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_stride32 = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_stride32, start, stop));
    printf("Kernel stride 32 time: %f ms\n", ms_stride32);

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_out1, d_out1, size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out2, d_out2, size_bytes, cudaMemcpyDeviceToHost));

    // Verify correctness
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_out1[i] != h_in[i] || h_out2[i] != h_in[i]) {
            ok = false;
            printf("Mismatch at %d: out1=%f, out2=%f, in=%f\n",
                   i, h_out1[i], h_out2[i], h_in[i]);
            break;
        }
    }
    if (ok) printf("Verification passed.\n");
    else    printf("Verification failed.\n");

    // Cleanup
    free(h_in);
    free(h_out1);
    free(h_out2);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out1));
    CUDA_CHECK(cudaFree(d_out2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```