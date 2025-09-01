/*
Aim of the program: Write a short paragraph in comments explaining the benefits of overlapping data transfers and computation.

The benefits of overlapping data transfers (host ↔ device) with computation on the GPU include:
- Hiding the latency of memory copies by performing them concurrently with kernel execution.
- Increasing overall throughput by keeping both the PCIe bus and the GPU compute units busy simultaneously.
- Reducing the total execution time for data‑parallel workloads, especially when the transfer size is large compared to kernel runtime.
- Improving resource utilization on systems with multiple GPUs or concurrent streams.

Thinking process:
- The prompt requires a single .cu file with a multiline comment at the top that states the aim of the program verbatim, includes the paragraph explaining benefits, and contains my reasoning steps.
- I decided to write a minimal CUDA example that uses two streams: one for host‑to‑device transfer and another for kernel execution, to illustrate overlap.
- I also added a simple kernel that multiplies each element by 2.
- I used cudaMemcpyAsync and launch the kernel on different streams to demonstrate asynchronous operation.
- I synchronise at the end to ensure all work is finished before freeing resources.
- The code is written in C (not C++), so I include <stdio.h> and <cuda_runtime.h>.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void multiply_by_two(const float *in, float *out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = in[idx] * 2.0f;
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    float *h_in  = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    // Initialize host input
    for (int i = 0; i < N; ++i)
        h_in[i] = (float)i;

    float *d_in  = NULL;
    float *d_out = NULL;

    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc((void**)&d_in,  size);
    if (err != cudaSuccess) { printf("cudaMalloc d_in failed\n"); return -1; }
    err = cudaMalloc((void**)&d_out, size);
    if (err != cudaSuccess) { printf("cudaMalloc d_out failed\n"); return -1; }

    // Create two streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Asynchronously copy input from host to device on stream1
    err = cudaMemcpyAsync(d_in, h_in, size, cudaMemcpyHostToDevice, stream1);
    if (err != cudaSuccess) { printf("cudaMemcpyAsync H2D failed\n"); return -1; }

    // Launch kernel on stream2 (will overlap with the transfer if possible)
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    multiply_by_two<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_in, d_out, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("Kernel launch failed\n"); return -1; }

    // Copy result back to host asynchronously on stream2 (after kernel completes)
    err = cudaMemcpyAsync(h_out, d_out, size, cudaMemcpyDeviceToHost, stream2);
    if (err != cudaSuccess) { printf("cudaMemcpyAsync D2H failed\n"); return -1; }

    // Wait for all operations to complete
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Simple validation
    int correct = 1;
    for (int i = 0; i < N; ++i)
    {
        if (h_out[i] != h_in[i] * 2.0f)
        {
            correct = 0;
            printf("Mismatch at index %d: got %f expected %f\n", i, h_out[i], h_in[i] * 2.0f);
            break;
        }
    }
    if (correct)
        printf("Result is correct.\n");

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
