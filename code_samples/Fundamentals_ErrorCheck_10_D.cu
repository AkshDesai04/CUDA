/*
Use your GpuTimer class to time a vector addition kernel.

Thought Process:
1. The task requires a CUDA program that demonstrates measuring the execution time of a simple
   vector addition kernel using a custom `GpuTimer` class.  
2. A minimal yet complete example should:
   - Allocate two input vectors and one output vector on the host.
   - Transfer them to the device.
   - Launch a vector addition kernel.
   - Time only the kernel execution (and synchronization to ensure completion).
   - Copy the result back and optionally verify it.
3. The `GpuTimer` wrapper will be built around CUDA events (`cudaEvent_t`) to provide high‑resolution timing.
4. We’ll include basic error checking with a macro `CUDA_CHECK` to simplify the code.
5. The kernel will be a straightforward element‑wise addition: `c[i] = a[i] + b[i]`.
6. The main function will:
   - Define the vector size (e.g., 1<<20 elements).
   - Initialize input vectors with some deterministic values.
   - Allocate device memory, copy data, create the timer, launch kernel, stop timer.
   - Print elapsed milliseconds.
   - Clean up all resources.
7. The program is self‑contained and can be compiled with `nvcc`:

   ```
   nvcc -o vector_add_timer vector_add_timer.cu
   ./vector_add_timer
   ```

   It will output the measured kernel execution time.

8. No external dependencies other than CUDA runtime are required. The code is written in standard C++ with CUDA extensions.
*/

#include <cuda_runtime.h>
#include <iostream>

// Simple macro for CUDA error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                      << " : " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// GpuTimer class for timing CUDA kernels
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }

    // Record start event
    void start() {
        CUDA_CHECK(cudaEventRecord(start_, 0));
    }

    // Record stop event and synchronize
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    // Return elapsed time in milliseconds
    float elapsed_ms() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// Vector addition kernel
__global__
void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Host allocations
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }

    // Device allocations
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Configure kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Timing the kernel
    GpuTimer timer;
    timer.start();
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    timer.stop();

    // Ensure kernel has finished
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness for a few elements
    bool correct = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": "
                      << h_c[i] << " != " << expected << std::endl;
            break;
        }
    }
    if (correct) {
        std::cout << "Vector addition verified for first 10 elements." << std::endl;
    }

    std::cout << "Kernel execution time: " << timer.elapsed_ms() << " ms" << std::endl;

    // Clean up
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
