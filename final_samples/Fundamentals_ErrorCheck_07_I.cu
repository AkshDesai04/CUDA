/*
Why does CUDA use an asynchronous model? (Performance, allows CPU and GPU to work in parallel).

Thinking:
To demonstrate the asynchronous execution model of CUDA, the program below performs
vector addition on the GPU while the CPU simultaneously does independent work
(a simple reduction).  It uses pinned host memory and a CUDA stream to enable
asynchronous memory copies and kernel launch.  The program measures the total
time taken and prints both the GPU result and the CPU‑side result.  This
illustrates how the CPU can progress while the GPU is busy, a key benefit of
CUDA’s asynchronous design.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

// Error checking macro
#define CUDA_CHECK(err)                                                   \
    do {                                                                  \
        cudaError_t err__ = (err);                                        \
        if (err__ != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA Error: %s (%d) at %s:%d\n",             \
                    cudaGetErrorString(err__), err__, __FILE__, __LINE__);\
            exit(EXIT_FAILURE);                                          \
        }                                                                 \
    } while (0)

// Simple vector addition kernel
__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

// CPU reduction (sum of elements)
float cpuReduce(const float *data, int N)
{
    float sum = 0.0f;
    for (int i = 0; i < N; ++i)
        sum += data[i];
    return sum;
}

int main(void)
{
    const int N = 1 << 24;                 // ~16M elements
    const size_t size = N * sizeof(float);

    // Allocate pinned host memory for async copy
    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost((void **)&h_A, size));
    CUDA_CHECK(cudaMallocHost((void **)&h_B, size));
    CUDA_CHECK(cudaMallocHost((void **)&h_C, size));

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;      // arbitrary value
        h_B[i] = 2.0f;      // arbitrary value
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Create stream for asynchronous operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Asynchronously copy inputs to device
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

    // Launch kernel asynchronously
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);

    // Asynchronously copy result back
    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));

    // CPU work: compute sum of input arrays concurrently
    float cpuSumA = cpuReduce(h_A, N);
    float cpuSumB = cpuReduce(h_B, N);
    float cpuTotal = cpuSumA + cpuSumB;

    // Wait for GPU operations to finish
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Record stop
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Verify GPU result for a few elements
    int errors = 0;
    for (int i = 0; i < 5; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            errors++;
            printf("Mismatch at %d: GPU=%f, Expected=%f\n", i, h_C[i], expected);
        }
    }

    printf("GPU vector addition completed in %.3f ms.\n", ms);
    printf("CPU reduction result (sum of A + sum of B): %.2f\n", cpuTotal);
    printf("GPU result sample: C[0]=%.2f, C[1]=%.2f, C[2]=%.2f\n",
           h_C[0], h_C[1], h_C[2]);

    if (errors == 0)
        printf("GPU result verified successfully.\n");
    else
        printf("GPU result has %d errors.\n", errors);

    // Clean up
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    return 0;
}
