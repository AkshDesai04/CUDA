```cpp
/*
Theoretical maximum speedup depends on the relative times of Copy, Exec, and Copy.
If T_copy_HtoD + T_copy_DtoH > T_exec, the program is memory-bound and overlap is very effective.

=== Thinking and Design Notes ===
1. The program demonstrates asynchronous data transfer and kernel execution using CUDA streams.
2. We measure three critical timing components:
   - T_copy_HtoD: time to copy input data from host to device.
   - T_exec: time to execute the kernel on the device.
   - T_copy_DtoH: time to copy output data from device back to host.
3. We launch the kernel in a separate CUDA stream from the memory copies to allow overlap.
4. Timing is measured with CUDA events (cudaEventRecord, cudaEventSynchronize, cudaEventElapsedTime).
5. After obtaining the raw timings (in milliseconds), we calculate:
   - Sequential time: T_seq = T_copy_HtoD + T_exec + T_copy_DtoH.
   - Overlapped time: T_ovlp = max(T_copy_HtoD, T_exec) + T_copy_DtoH.
   - Theoretical speedup: Speedup = T_seq / T_ovlp.
6. The program prints the three timings, the overlapped time, the sequential time, and the speedup.
7. By varying array size or kernel workload, we can observe the memory-bound scenario
   where T_copy_HtoD + T_copy_DtoH > T_exec, leading to an effective overlap.
8. The kernel performs a simple vector addition (C = A + B) to keep the example focused
   on transfer overhead rather than computation.
9. The code is self-contained, written in CUDA C, and can be compiled with nvcc:
   nvcc -o async_overlap async_overlap.cu
10. No external libraries are required beyond the CUDA runtime.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Simple vector addition kernel: C = A + B
__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// Helper function to check CUDA errors
inline void checkCuda(cudaError_t result, const char *msg)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    // Parameters
    const int N = 1 << 24;               // 16M elements (~64 MB per array)
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc((void **)&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void **)&d_B, bytes), "cudaMalloc d_B");
    checkCuda(cudaMalloc((void **)&d_C, bytes), "cudaMalloc d_C");

    // Create CUDA streams
    cudaStream_t streamCopy, streamKernel;
    checkCuda(cudaStreamCreate(&streamCopy), "cudaStreamCreate streamCopy");
    checkCuda(cudaStreamCreate(&streamKernel), "cudaStreamCreate streamKernel");

    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    checkCuda(cudaEventCreate(&startEvent), "cudaEventCreate startEvent");
    checkCuda(cudaEventCreate(&stopEvent), "cudaEventCreate stopEvent");

    // --- Asynchronous data transfer and kernel launch with overlap ---

    // Record start event
    checkCuda(cudaEventRecord(startEvent, 0), "cudaEventRecord startEvent");

    // 1. Asynchronously copy A and B to device in streamCopy
    checkCuda(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, streamCopy), "cudaMemcpyAsync HtoD d_A");
    checkCuda(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, streamCopy), "cudaMemcpyAsync HtoD d_B");

    // 2. Launch kernel in streamKernel (will wait for device to be ready)
    checkCuda(cudaMemcpyAsync(NULL, NULL, 0, cudaMemcpyDefault, streamKernel), "cudaStreamSynchronize dummy");
    checkCuda(cudaEventRecord(startEvent, streamKernel), "cudaEventRecord startEvent kernel");
    vecAdd<<<blocksPerGrid, threadsPerBlock, 0, streamKernel>>>(d_A, d_B, d_C, N);
    checkCuda(cudaGetLastError(), "Kernel launch");

    // 3. Asynchronously copy result back to host in streamCopy after kernel completes
    checkCuda(cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, streamCopy), "cudaMemcpyAsync DtoH d_C");

    // Record stop event
    checkCuda(cudaEventRecord(stopEvent, streamCopy), "cudaEventRecord stopEvent");

    // Synchronize to ensure all operations are finished
    checkCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize stopEvent");

    // Measure elapsed time in milliseconds
    float ms_total;
    checkCuda(cudaEventElapsedTime(&ms_total, startEvent, stopEvent), "cudaEventElapsedTime");

    // --- Separate measurements for individual components ---

    // Measure HtoD copy time
    cudaEvent_t h2d_start, h2d_stop;
    checkCuda(cudaEventCreate(&h2d_start), "cudaEventCreate h2d_start");
    checkCuda(cudaEventCreate(&h2d_stop), "cudaEventCreate h2d_stop");
    checkCuda(cudaEventRecord(h2d_start, 0), "cudaEventRecord h2d_start");
    checkCuda(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, streamCopy), "cudaMemcpyAsync HtoD d_A");
    checkCuda(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, streamCopy), "cudaMemcpyAsync HtoD d_B");
    checkCuda(cudaEventRecord(h2d_stop, streamCopy), "cudaEventRecord h2d_stop");
    checkCuda(cudaEventSynchronize(h2d_stop), "cudaEventSynchronize h2d_stop");
    float ms_h2d;
    checkCuda(cudaEventElapsedTime(&ms_h2d, h2d_start, h2d_stop), "cudaEventElapsedTime h2d");

    // Measure kernel execution time
    cudaEvent_t k_start, k_stop;
    checkCuda(cudaEventCreate(&k_start), "cudaEventCreate k_start");
    checkCuda(cudaEventCreate(&k_stop), "cudaEventCreate k_stop");
    checkCuda(cudaEventRecord(k_start, 0), "cudaEventRecord k_start");
    vecAdd<<<blocksPerGrid, threadsPerBlock, 0, streamKernel>>>(d_A, d_B, d_C, N);
    checkCuda(cudaEventRecord(k_stop, streamKernel), "cudaEventRecord k_stop");
    checkCuda(cudaEventSynchronize(k_stop), "cudaEventSynchronize k_stop");
    float ms_k;
    checkCuda(cudaEventElapsedTime(&ms_k, k_start, k_stop), "cudaEventElapsedTime k");

    // Measure DtoH copy time
    cudaEvent_t d2h_start, d2h_stop;
    checkCuda(cudaEventCreate(&d2h_start), "cudaEventCreate d2h_start");
    checkCuda(cudaEventCreate(&d2h_stop), "cudaEventCreate d2h_stop");
    checkCuda(cudaEventRecord(d2h_start, 0), "cudaEventRecord d2h_start");
    checkCuda(cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, streamCopy), "cudaMemcpyAsync DtoH d_C");
    checkCuda(cudaEventRecord(d2h_stop, streamCopy), "cudaEventRecord d2h_stop");
    checkCuda(cudaEventSynchronize(d2h_stop), "cudaEventSynchronize d2h_stop");
    float ms_d2h;
    checkCuda(cudaEventElapsedTime(&ms_d2h, d2h_start, d2h_stop), "cudaEventElapsedTime d2h");

    // Compute sequential and overlapped times
    float seq_time = ms_h2d + ms_k + ms_d2h;
    float ovlp_time = fmaxf(ms_h2d, ms_k) + ms_d2h;
    float speedup = seq_time / ovlp_time;

    // Output results
    printf("=== Timing Results (ms) ===\n");
    printf("Host-to-Device copy time:   %f\n", ms_h2d);
    printf("Kernel execution time:      %f\n", ms_k);
    printf("Device-to-Host copy time:   %f\n", ms_d2h);
    printf("\nSequential time:            %f\n", seq_time);
    printf("Overlapped time:            %f\n", ovlp_time);
    printf("Theoretical speedup:        %f\n", speedup);

    // Clean up
    checkCuda(cudaEventDestroy(startEvent), "cudaEventDestroy startEvent");
    checkCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy stopEvent");
    checkCuda(cudaEventDestroy(h2d_start), "cudaEventDestroy h2d_start");
    checkCuda(cudaEventDestroy(h2d_stop), "cudaEventDestroy h2d_stop");
    checkCuda(cudaEventDestroy(k_start), "cudaEventDestroy k_start");
    checkCuda(cudaEventDestroy(k_stop), "cudaEventDestroy k_stop");
    checkCuda(cudaEventDestroy(d2h_start), "cudaEventDestroy d2h_start");
    checkCuda(cudaEventDestroy(d2h_stop), "cudaEventDestroy d2h_stop");

    checkCuda(cudaStreamDestroy(streamCopy), "cudaStreamDestroy streamCopy");
    checkCuda(cudaStreamDestroy(streamKernel), "cudaStreamDestroy streamKernel");

    checkCuda(cudaFree(d_A), "cudaFree d_A");
    checkCuda(cudaFree(d_B), "cudaFree d_B");
    checkCuda(cudaFree(d_C), "cudaFree d_C");

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```