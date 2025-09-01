/*
Aim: Compare the timing from CUDA events to timing with `std::chrono::high_resolution_clock`. Explain why CUDA events are more accurate for timing GPU operations.
The program launches a simple vector addition kernel and measures its execution time using two different approaches:
1. `std::chrono::high_resolution_clock` on the host side.
2. CUDA event timing which measures GPU wall‑clock time directly.
We allocate host and device memory, initialize data, and run the kernel twice – once for each timing method.
The chrono measurement is performed before the kernel launch and after `cudaDeviceSynchronize()` to ensure the kernel has finished,
but this introduces CPU‑side overhead and synchronization latency that is not part of the actual GPU work.
The CUDA event measurement uses two events recorded on the same stream; the elapsed time is reported by the GPU’s timestamp
hardware and is immune to CPU scheduling delays. The program prints both timings and includes explanatory comments in the output.
*/

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main()
{
    const int N = 1 << 20;          // 1M elements
    const int bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // ---- Timing with std::chrono ----
    auto start_cpu = std::chrono::high_resolution_clock::now();

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for kernel to finish to get a correct measurement
    cudaDeviceSynchronize();

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_cpu = end_cpu - start_cpu;

    std::cout << "CPU chrono measured time: " << elapsed_cpu.count() << " ms\n";

    // ---- Timing with CUDA events ----
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, 0);                          // Record start
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop_event, 0);                           // Record stop

    // Wait for the event to complete
    cudaEventSynchronize(stop_event);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);

    std::cout << "CUDA event measured time: " << elapsed_ms << " ms\n";

    // Clean up
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    // Explanation (printed as comments)
    std::cout << "\nExplanation:\n";
    std::cout << "1. std::chrono measures CPU wall‑clock time. Since the kernel launch is asynchronous, "
                 "we must call cudaDeviceSynchronize() to ensure the kernel has finished before taking the "
                 "end time. This synchronization adds overhead and includes CPU scheduling delays.\n";
    std::cout << "2. CUDA events are recorded on the GPU itself. The elapsed time reported by cudaEventElapsedTime() "
                 "comes from the GPU's internal timestamp counters, giving an accurate measurement of the "
                 "GPU's execution time only, without CPU overhead.\n";
    std::cout << "Therefore, CUDA events provide a more accurate timing for GPU operations.\n";

    return 0;
}