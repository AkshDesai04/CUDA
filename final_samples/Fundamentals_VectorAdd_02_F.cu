/*
Aim: Compare the execution time for `N=1M` vs `N=2M`. Is it roughly double?

Thinking process:

1. I want a simple CUDA kernel that scales linearly with the number of elements, so that
   the execution time should ideally be proportional to N. A vector addition
   (`C[i] = A[i] + B[i]`) is a classic example that only does arithmetic on each
   element and has no branching or shared memory complexity.

2. To compare timings I will:
   - Allocate three arrays (A, B, C) on the host and device.
   - Initialize A and B with some values (e.g., sequential numbers) on the host.
   - Copy them to the device.
   - Launch the kernel for a given N and record the elapsed GPU time using
     CUDA events (`cudaEvent_t`).  I will run each size multiple times (e.g.
     10 iterations) and average the times to smooth out jitter.
   - After the kernel I will copy the result back (optional) to avoid
     potential lazy copying or hidden overheads.
   - I will print the average execution time in milliseconds for each N.
   - Finally I will compute the ratio `time_N2M / time_N1M` and output it.

3. Expected behavior:
   The time for N=2M should be roughly twice the time for N=1M, although
   kernel launch overhead and memory transfer times may introduce a small
   deviation.

4. Error handling: I will use a simple macro to check CUDA API calls and
   kernel launch errors to make the code robust.

5. The program is self‑contained, uses only the CUDA runtime API, and
   should compile with `nvcc`.  It prints the timings and the ratio,
   allowing the user to answer the question in the prompt.

*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

__global__ void vectorAdd(const float* A, const float* B, float* C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main()
{
    const size_t N1 = 1'000'000;   // 1M elements
    const size_t N2 = 2'000'000;   // 2M elements

    const int iterations = 10;    // number of runs to average

    // Host pointers
    float *h_A1 = nullptr, *h_B1 = nullptr, *h_C1 = nullptr;
    float *h_A2 = nullptr, *h_B2 = nullptr, *h_C2 = nullptr;

    // Device pointers
    float *d_A1 = nullptr, *d_B1 = nullptr, *d_C1 = nullptr;
    float *d_A2 = nullptr, *d_B2 = nullptr, *d_C2 = nullptr;

    // Allocate host memory
    h_A1 = (float*)malloc(N1 * sizeof(float));
    h_B1 = (float*)malloc(N1 * sizeof(float));
    h_C1 = (float*)malloc(N1 * sizeof(float));

    h_A2 = (float*)malloc(N2 * sizeof(float));
    h_B2 = (float*)malloc(N2 * sizeof(float));
    h_C2 = (float*)malloc(N2 * sizeof(float));

    if (!h_A1 || !h_B1 || !h_C1 || !h_A2 || !h_B2 || !h_C2) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (size_t i = 0; i < N1; ++i) {
        h_A1[i] = static_cast<float>(i);
        h_B1[i] = static_cast<float>(N1 - i);
    }

    for (size_t i = 0; i < N2; ++i) {
        h_A2[i] = static_cast<float>(i);
        h_B2[i] = static_cast<float>(N2 - i);
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_A1, N1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B1, N1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C1, N1 * sizeof(float)));

    CHECK_CUDA(cudaMalloc((void**)&d_A2, N2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B2, N2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C2, N2 * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A1, h_A1, N1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B1, h_B1, N1 * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_A2, h_A2, N2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B2, h_B2, N2 * sizeof(float), cudaMemcpyHostToDevice));

    // CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int threadsPerBlock = 256;

    // Timing for N1
    float totalTimeN1 = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        int blocksPerGrid1 = (N1 + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid1, threadsPerBlock>>>(d_A1, d_B1, d_C1, N1);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTimeN1 += milliseconds;
    }
    float avgTimeN1 = totalTimeN1 / iterations;

    // Timing for N2
    float totalTimeN2 = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        int blocksPerGrid2 = (N2 + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid2, threadsPerBlock>>>(d_A2, d_B2, d_C2, N2);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTimeN2 += milliseconds;
    }
    float avgTimeN2 = totalTimeN2 / iterations;

    // Optionally copy results back (not used further)
    CHECK_CUDA(cudaMemcpy(h_C1, d_C1, N1 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C2, d_C2, N2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    printf("Average GPU execution time over %d iterations:\n", iterations);
    printf("N = %zu elements: %.3f ms\n", N1, avgTimeN1);
    printf("N = %zu elements: %.3f ms\n", N2, avgTimeN2);
    printf("Ratio (N=2M / N=1M): %.3f\n", avgTimeN2 / avgTimeN1);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_A1));
    CHECK_CUDA(cudaFree(d_B1));
    CHECK_CUDA(cudaFree(d_C1));

    CHECK_CUDA(cudaFree(d_A2));
    CHECK_CUDA(cudaFree(d_B2));
    CHECK_CUDA(cudaFree(d_C2));

    free(h_A1);
    free(h_B1);
    free(h_C1);
    free(h_A2);
    free(h_B2);
    free(h_C2);

    return 0;
}
