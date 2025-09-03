```cuda
/* 
Aim: Demonstrate the performance benefit: write two kernels, one where each thread reads a configuration parameter from global memory, and one where it reads from constant memory. Time both. 

Thinking:
- A single configuration parameter (float) is stored in two locations: 
  1) a device global memory buffer (float *d_globalParam)
  2) a device constant memory symbol (d_constParam)
- Two kernels are written:
  * kernelGlobal reads the parameter from global memory (d_globalParam[0])
  * kernelConst reads the parameter from constant memory (d_constParam)
- Both kernels perform a trivial computation that uses the parameter to avoid the compiler eliminating the memory load:
  out[idx] = p * idx
- We allocate a large output array (N = 1<<24 ≈ 16M elements) so that the kernels have enough work to make the timing meaningful.
- Each kernel is launched multiple times (NUM_RUNS = 10) to average the execution time.
- CUDA events are used for precise timing, and cudaDeviceSynchronize() ensures completion before measuring the stop time.
- Results are copied back to host and a checksum is computed to ensure the kernels actually ran and were not optimized away.
- Error checking macro (CHECK) is used after each CUDA call for safety.
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#define CHECK(call)                                                         \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << " : " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__constant__ float d_constParam;

// Kernel that reads the configuration parameter from global memory
__global__ void kernelGlobal(float *out, const float *param, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float p = param[0];          // global memory read
        out[idx] = p * idx;          // dummy operation
    }
}

// Kernel that reads the configuration parameter from constant memory
__global__ void kernelConst(float *out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float p = d_constParam;      // constant memory read
        out[idx] = p * idx;          // dummy operation
    }
}

int main(int argc, char **argv)
{
    const size_t N = 1 << 24;          // 16 million elements
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int NUM_RUNS = 10;           // number of times each kernel is launched for averaging

    // Host allocations
    float *h_out = (float*)malloc(N * sizeof(float));
    if (!h_out) {
        std::cerr << "Host allocation failed" << std::endl;
        return EXIT_FAILURE;
    }

    // Device allocations
    float *d_out;
    float *d_globalParam;
    CHECK(cudaMalloc((void**)&d_out, N * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_globalParam, sizeof(float)));

    // Configuration parameter
    float hostParam = 3.1415926535f;

    // Copy to device global memory
    CHECK(cudaMemcpy(d_globalParam, &hostParam, sizeof(float), cudaMemcpyHostToDevice));

    // Copy to constant memory
    CHECK(cudaMemcpyToSymbol(d_constParam, &hostParam, sizeof(float), 0, cudaMemcpyHostToDevice));

    // Events for timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Variables to hold elapsed times
    float timeGlobal = 0.0f;
    float timeConst  = 0.0f;

    // Warm-up launch
    kernelGlobal<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_globalParam, N);
    kernelConst<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, N);
    CHECK(cudaDeviceSynchronize());

    // Timing kernelGlobal
    for (int run = 0; run < NUM_RUNS; ++run) {
        CHECK(cudaEventRecord(start, 0));
        kernelGlobal<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_globalParam, N);
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        timeGlobal += ms;
    }

    // Timing kernelConst
    for (int run = 0; run < NUM_RUNS; ++run) {
        CHECK(cudaEventRecord(start, 0));
        kernelConst<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, N);
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        timeConst += ms;
    }

    // Copy result back to host to prevent dead code elimination
    CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute simple checksum
    double checksum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        checksum += h_out[i];
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average time for kernelGlobal  : " << timeGlobal / NUM_RUNS << " ms" << std::endl;
    std::cout << "Average time for kernelConst   : " << timeConst  / NUM_RUNS << " ms" << std::endl;
    std::cout << "Checksum (to verify computation) : " << checksum << std::endl;

    // Clean up
    free(h_out);
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_globalParam));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```