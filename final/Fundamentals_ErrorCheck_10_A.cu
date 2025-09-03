/*
The destructor of the GpuTimer class should call `cudaEventDestroy` for both events to prevent resource leaks.

Thinking process:

1. Understand the requirement: We need to ensure that the GPU timer properly releases the CUDA events when the timer object is destroyed. This involves calling `cudaEventDestroy` on each event (start and stop).
2. Plan the structure:
   - Create a `GpuTimer` class that encapsulates two CUDA events: `m_startEvent` and `m_stopEvent`.
   - Provide methods `start()`, `stop()`, and `elapsed()` to record and compute elapsed time.
   - In the destructor, call `cudaEventDestroy` on both events.
3. Handle error checking:
   - Use a helper function `checkCuda(cudaError_t)` to simplify error handling.
   - Ensure that event creation and destruction handle potential errors.
4. Write a small example in `main()` that demonstrates the timer on a dummy kernel launch.
5. Compile the file with `nvcc`. No external dependencies beyond CUDA runtime.
6. Make sure the file is self-contained, compiles, and follows the requirement.

*/

#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <chrono>

/* Helper macro for checking CUDA errors */
inline void checkCuda(cudaError_t err, const char *msg = nullptr)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << (msg ? msg : "") << " " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/* GpuTimer class */
class GpuTimer
{
public:
    GpuTimer()
    {
        checkCuda(cudaEventCreate(&m_startEvent), "Failed to create start event");
        checkCuda(cudaEventCreate(&m_stopEvent), "Failed to create stop event");
        m_elapsedTime = 0.0f;
    }

    ~GpuTimer()
    {
        /* The destructor must destroy both CUDA events to release GPU resources. */
        checkCuda(cudaEventDestroy(m_startEvent), "Failed to destroy start event");
        checkCuda(cudaEventDestroy(m_stopEvent), "Failed to destroy stop event");
    }

    void start()
    {
        checkCuda(cudaEventRecord(m_startEvent, 0), "Failed to record start event");
    }

    void stop()
    {
        checkCuda(cudaEventRecord(m_stopEvent, 0), "Failed to record stop event");
        /* Wait for the stop event to complete */
        checkCuda(cudaEventSynchronize(m_stopEvent), "Failed to synchronize stop event");
        /* Compute elapsed time */
        checkCuda(cudaEventElapsedTime(&m_elapsedTime, m_startEvent, m_stopEvent),
                  "Failed to compute elapsed time");
    }

    float elapsedMilliseconds() const { return m_elapsedTime; }

private:
    cudaEvent_t m_startEvent;
    cudaEvent_t m_stopEvent;
    float m_elapsedTime;
};

/* Dummy kernel for demonstration */
__global__ void dummyKernel(float *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // Simple computation to consume some time
        data[idx] = __sinf(data[idx]) * __cosf(data[idx]) + __expf(data[idx]);
    }
}

int main()
{
    const int N = 1 << 20;            // 1M elements
    const size_t size = N * sizeof(float);

    float *d_data = nullptr;
    checkCuda(cudaMalloc(&d_data, size), "Failed to allocate device memory");

    /* Launch kernel and time it */
    GpuTimer timer;
    timer.start();

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    dummyKernel<<<blocks, threadsPerBlock>>>(d_data, N);

    timer.stop();

    std::cout << "Kernel execution time: " << timer.elapsedMilliseconds() << " ms" << std::endl;

    checkCuda(cudaFree(d_data), "Failed to free device memory");

    return 0;
}
