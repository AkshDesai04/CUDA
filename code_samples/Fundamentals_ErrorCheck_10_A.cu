/*
The destructor of the `GpuTimer` class should call `cudaEventDestroy` for both events to prevent resource leaks.

**Thinking**  
1. The user specifically wants the destructor of `GpuTimer` to destroy both CUDA events.  
2. To satisfy that requirement, the class will create two `cudaEvent_t` members: one for the start event and one for the stop event.  
3. In the constructor, both events are created with `cudaEventCreate`.  
4. In the destructor, `cudaEventDestroy` is called for each event to clean up resources.  
5. The class will provide `start()`, `stop()`, and `elapsed()` methods for measuring GPU execution time.  
6. A simple example kernel (`vecAdd`) and a main function are provided to demonstrate usage of `GpuTimer`.  
7. The code is written in C++ style (CUDA supports C++), compiled with `nvcc`.  
8. No external dependencies other than the CUDA runtime are needed.  
9. The program is self‑contained: it allocates memory, launches a kernel, times it, prints the elapsed time, and cleans up all resources, including destroying the CUDA events in the destructor.

**Note**: Only the .cu file contents are output below; no additional explanatory text. */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
    }

    void start() {
        cudaEventRecord(start_event_, 0);
    }

    void stop() {
        cudaEventRecord(stop_event_, 0);
        cudaEventSynchronize(stop_event_);
    }

    float elapsed() const {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_event_, stop_event_);
        return ms;
    }

private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
};

__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char* argv[]) {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = sinf(i);
        h_B[i] = cosf(i);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    GpuTimer timer;
    timer.start();

    vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    timer.stop();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("GPU time for vecAdd: %f ms\n", timer.elapsed());

    // Verify a few elements
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) {
            ok = false;
            break;
        }
    }
    printf("Verification: %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
