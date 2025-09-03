/*
Use `nsight compute` or `nvprof` to measure the global memory bandwidth and transaction count for the naive kernel. Note the high number of transactions.

Thinking:
- We need a CUDA program that launches a kernel with intentionally non-coalesced memory accesses to produce many global memory transactions.
- The kernel will read from a global array with a stride that forces each thread to access memory far apart, causing many small memory transactions.
- Allocate large enough arrays to get measurable bandwidth.
- Provide timing using cudaEvent for kernel duration.
- Provide instructions in comments on how to run nvprof or nsight compute to capture metrics: e.g. `nvprof --metrics global_load_transactions,global_load_bytes <executable>`.
- The code should be compilable with nvcc and produce an executable that can be profiled.
- The naive kernel: each thread loads one element from `input[threadIdx + blockIdx.x * blockDim.x]`? That is coalesced. To make it naive, use a stride: e.g., `int stride = 16; int inIdx = idx * stride;` ensuring out-of-bounds is avoided.
- We can set N=1<<20 and stride=16, so input size is N*stride. Use floats.
- We'll compute output[idx] = input[inIdx] * 2.0f.
- After kernel, copy back output and maybe print sum to avoid compiler optimizing away.
- Use cudaMallocManaged or separate device memory.
- We will not include any external profiling calls in code; just rely on nvprof/NSight.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

const int N = 1 << 20;      // Number of output elements
const int STRIDE = 16;      // Stride to create uncoalesced accesses

__global__ void naive_kernel(const float *input, float *output, int n, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int inIdx = idx * stride;
        output[idx] = input[inIdx] * 2.0f;
    }
}

int main(void)
{
    const int inputSize = N * STRIDE;
    const int outputSize = N;
    const size_t inputBytes = inputSize * sizeof(float);
    const size_t outputBytes = outputSize * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(inputBytes);
    float *h_output = (float *)malloc(outputBytes);

    // Initialize input
    for (int i = 0; i < inputSize; ++i) {
        h_input[i] = 1.0f;  // simple value, could be random
    }

    // Allocate device memory
    float *d_input = NULL;
    float *d_output = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_input, inputBytes));
    CHECK_CUDA(cudaMalloc((void **)&d_output, outputBytes));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Launch kernel
    CHECK_CUDA(cudaEventRecord(start, 0));
    naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, STRIDE);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Measure elapsed time
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, outputBytes, cudaMemcpyDeviceToHost));

    // Simple reduction to avoid compiler optimization
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += h_output[i];
    }
    printf("Result sum = %f\n", sum);
    printf("Kernel execution time = %f ms\n", ms);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);

    /*
    To profile this program using nvprof, compile with:
        nvcc -o naive_profiler naive_profiler.cu

    Then run:
        nvprof --metrics global_load_transactions,global_load_bytes,global_store_transactions,global_store_bytes ./naive_profiler

    To use Nsight Compute, launch from the command line or launch Nsight Compute GUI and attach to the running executable.

    The metrics of interest:
        - global_load_transactions: number of global memory load transactions
        - global_load_bytes: bytes transferred from global memory
        - global_store_transactions: number of global memory store transactions
        - global_store_bytes: bytes transferred to global memory

    Because of the intentional stride, you should observe a high number of load transactions and relatively low bandwidth per transaction.
    */

    return 0;
}
