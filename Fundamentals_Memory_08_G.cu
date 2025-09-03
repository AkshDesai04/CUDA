/*
Aim of the program: 
Compare the timing from CUDA events to timing with `std::chrono::high_resolution_clock`. 
Explain why CUDA events are more accurate for timing GPU operations.

My entire thinking: 
- The goal is to demonstrate a practical comparison between host-side high-resolution clock timing and device-side CUDA event timing for a GPU kernel. 
- I will write a simple kernel (vector addition) that can be launched many times to produce a measurable duration. 
- For host timing, I will record the time before launching the kernel, then call `cudaDeviceSynchronize()` to ensure the kernel finishes, then record the end time using `std::chrono::high_resolution_clock`. The difference gives the total time including host overhead. 
- For CUDA event timing, I will create two events (`start` and `stop`), record the start event just before kernel launch, record the stop event immediately after launch, then call `cudaEventSynchronize(stop)` to wait for the kernel to finish. The elapsed time measured by `cudaEventElapsedTime` is device time only, unaffected by host scheduling or overhead. 
- Because CUDA events are recorded on the GPU timeline, they provide nanosecond resolution and exclude host-side scheduling, thread dispatch, and other system overhead. `std::chrono::high_resolution_clock` measures wall-clock time, which includes all those factors and may be less precise due to scheduling and context switches. 
- In the output, I will print both timings and the difference, illustrating why event timing is often more reliable for GPU performance measurement. 
- The code will also handle error checking, simple argument parsing, and clean up resources. 
- Comments in the code will explain each step. 
*/

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " : " \
                      << cudaGetErrorString(err) << std::endl;         \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main() {
    const int N = 1 << 20; // 1M elements
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // ------------------ Host timing with std::chrono ------------------
    auto host_start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Synchronize to ensure kernel has finished
    CUDA_CHECK(cudaDeviceSynchronize());

    auto host_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> host_elapsed = host_end - host_start;

    // ------------------ Device timing with CUDA events ------------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float device_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&device_ms, start, stop));

    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify correctness (optional)
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) {
            correct = false;
            break;
        }
    }
    if (!correct) {
        std::cerr << "Result verification failed!" << std::endl;
    }

    // Print timings
    std::cout << "Host timing (chrono):   " << host_elapsed.count() << " ms" << std::endl;
    std::cout << "Device timing (CUDA ev): " << device_ms << " ms" << std::endl;
    std::cout << "Difference:             " << host_elapsed.count() - device_ms << " ms" << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
