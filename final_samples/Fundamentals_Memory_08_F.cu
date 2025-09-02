/*
Wrap the event timer logic in a C++ class `GpuTimer` for convenience.

The aim of this program is to provide a reusable CUDA timer class that
encapsulates CUDA event creation, recording, and elapsed time calculation.
Using CUDA events is the recommended way to time GPU code because they
are recorded on the device and provide accurate measurements of the time
elapsed between two points on the GPU.  
We need to support:
- Construction: create start and stop events.
- Destruction: destroy the events to avoid leaks.
- start(): record the start event.
- stop(): record the stop event.
- elapsed(): compute the elapsed time in milliseconds between start and stop.
  
The class will use the CUDA Runtime API (`cudaEventCreateWithFlags`,
`cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`).
It will also check for errors after each CUDA call and abort if an error
occurs, printing an informative message.

To demonstrate the class, a simple dummy kernel is launched and timed.
The kernel just performs some arithmetic on an array.  The main function
shows how to use `GpuTimer` to time the kernel launch and the subsequent
device-to-host copy.

The code is self‑contained and can be compiled with `nvcc`:
    nvcc -o GpuTimer GpuTimer.cu
Then run:
    ./GpuTimer
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                              \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

// GpuTimer class definition
class GpuTimer {
public:
    GpuTimer() {
        // Create events with default flags
        CUDA_CHECK(cudaEventCreateWithFlags(&start_, cudaEventDefault));
        CUDA_CHECK(cudaEventCreateWithFlags(&stop_, cudaEventDefault));
    }

    ~GpuTimer() {
        // Destroy events
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }

    // Record start event
    void start() {
        CUDA_CHECK(cudaEventRecord(start_, 0));
    }

    // Record stop event
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_, 0));
    }

    // Wait for stop event to complete and return elapsed time in milliseconds
    float elapsed() {
        // Wait until the stop event has completed
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

// Dummy kernel: simple vector addition
__global__ void dummyKernel(float* d_out, const float* d_in1, const float* d_in2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = d_in1[idx] + d_in2[idx];
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_in1 = (float*)malloc(size);
    float *h_in2 = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_in1[i] = static_cast<float>(i);
        h_in2[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_in1, *d_in2, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in1, size));
    CUDA_CHECK(cudaMalloc(&d_in2, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_in1, h_in1, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in2, h_in2, size, cudaMemcpyHostToDevice));

    // Create timer
    GpuTimer timer;

    // Determine block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Start timing
    timer.start();

    // Launch kernel
    dummyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in1, d_in2, N);
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Stop timing
    timer.stop();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Wait for all operations to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get elapsed time
    float ms = timer.elapsed();
    printf("Kernel execution time: %.3f ms\n", ms);

    // Cleanup
    CUDA_CHECK(cudaFree(d_in1));
    CUDA_CHECK(cudaFree(d_in2));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in1);
    free(h_in2);
    free(h_out);

    return 0;
}
