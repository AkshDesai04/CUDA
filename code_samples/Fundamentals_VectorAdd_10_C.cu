```cpp
/*
Plot a graph of performance (elements/sec) vs. vector size (N) for both CPU and GPU. Find the crossover point where the GPU becomes faster.

**Thinking and Approach**  
The task is to measure the throughput of a simple element-wise vector operation (addition) on both CPU and GPU across a range of vector sizes, and determine at which vector size the GPU outperforms the CPU.  
Key points in designing this program:

1. **Operation**: Vector addition is chosen because it is compute-bound and requires minimal memory overhead, making it a good candidate for comparing CPU and GPU performance.

2. **Measurement**:
   - CPU timing will be performed with `std::chrono::high_resolution_clock`.  
   - GPU timing will use CUDA events (`cudaEventRecord` and `cudaEventElapsedTime`).

3. **Vector Size Range**: A sequence of powers of two from \(2^{10}\) to \(2^{24}\) (i.e., 1024 to 16,777,216 elements) provides a broad spectrum of workloads. This range is sufficient to capture the crossover point.

4. **Memory Allocation**:  
   - Host arrays are allocated once with the maximum size.  
   - Device arrays are allocated once with the maximum size to avoid repeated allocation overhead.  
   - For each size \(N\), only the first \(N\) elements are used; the rest remain untouched.

5. **Kernel Configuration**: A simple kernel adds two input vectors into an output vector. The block size is fixed at 256 threads, and the grid size is computed to cover all \(N\) elements.

6. **Throughput Calculation**: Throughput is calculated as `elements / seconds`. We keep the values for both CPU and GPU for each \(N\) and output them in CSV format: `N,CPU_throughput,GPU_throughput`.

7. **Crossover Detection**: After all measurements, the program scans the recorded throughputs and finds the smallest \(N\) where GPU throughput exceeds CPU throughput. That \(N\) is printed as the crossover point.

8. **Error Checking**: All CUDA API calls are wrapped in a simple macro to check for errors.

9. **Plotting**: The program does not directly plot; instead, it outputs CSV data that can be piped into any external plotting tool (e.g., GNUplot, matplotlib).

The resulting .cu file compiles with `nvcc` and produces the required measurements and crossover point.

*/

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error (" << err << "): " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void vecAddKernel(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void cpuVecAdd(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int minExp = 10;          // 2^10 = 1024
    const int maxExp = 24;          // 2^24 = 16,777,216
    const int maxN   = 1 << maxExp; // maximum vector size

    // Allocate host memory
    float *h_A = new float[maxN];
    float *h_B = new float[maxN];
    float *h_C = new float[maxN]; // result

    // Random number generator
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, maxN * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, maxN * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, maxN * sizeof(float)));

    // Prepare CSV header
    std::cout << "N,CPU_throughput_GHz,GPU_throughput_GHz" << std::endl;

    // Vectors to store throughputs for crossover detection
    std::vector<int> sizes;
    std::vector<double> cpuThroughputs;
    std::vector<double> gpuThroughputs;

    const int threadsPerBlock = 256;

    for (int exp = minExp; exp <= maxExp; ++exp) {
        int N = 1 << exp;
        sizes.push_back(N);

        // Initialize host data
        for (int i = 0; i < N; ++i) {
            h_A[i] = dist(rng);
            h_B[i] = dist(rng);
        }

        // CPU timing
        auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuVecAdd(h_A, h_B, h_C, N);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpuDuration = cpuEnd - cpuStart;
        double cpuTimeSec = cpuDuration.count();
        double cpuThroughput = (double)N / cpuTimeSec / 1e9; // elements per second in GHz
        cpuThroughputs.push_back(cpuThroughput);

        // GPU timing
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

        cudaEvent_t startEvent, stopEvent;
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));

        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

        CUDA_CHECK(cudaEventRecord(startEvent, 0));
        vecAddKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stopEvent, 0));

        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        float gpuMillis = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpuMillis, startEvent, stopEvent));
        double gpuTimeSec = gpuMillis / 1000.0; // convert ms to s
        double gpuThroughput = (double)N / gpuTimeSec / 1e9;
        gpuThroughputs.push_back(gpuThroughput);

        // Cleanup events
        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(stopEvent));

        // Output CSV line
        std::cout << N << ","
                  << std::fixed << std::setprecision(3) << cpuThroughput << ","
                  << std::fixed << std::setprecision(3) << gpuThroughput << std::endl;
    }

    // Find crossover point
    int crossoverN = -1;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (gpuThroughputs[i] > cpuThroughputs[i]) {
            crossoverN = sizes[i];
            break;
        }
    }

    if (crossoverN != -1) {
        std::cout << "\nCrossover point: N = " << crossoverN << " elements." << std::endl;
    } else {
        std::cout << "\nNo crossover point found within the tested range." << std::endl;
    }

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```