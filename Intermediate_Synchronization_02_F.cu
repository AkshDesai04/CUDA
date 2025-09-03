```cpp
/*********************************************************************
 * Aim of the program:
 * Use atomicMax to implement a single-kernel max reduction and compare
 * its performance to the two-pass method.
 *
 * My thoughts while writing this code:
 * 1. We need two different reduction strategies:
 *    a. A single kernel where every thread calls atomicMax on a global
 *       variable. This is simple but may have contention.
 *    b. A two-pass method:
 *       - First pass: each block reduces its chunk into a per-block
 *         maximum in shared memory and writes it to a global array.
 *       - Second pass: a second kernel (or a loop on CPU) reduces the
 *         per-block maxima to the final result. This reduces contention
 *         because the atomic operation is only performed once per block.
 *
 * 2. Timing:
 *    Use cudaEvent_t to time both kernels separately. We'll also
 *    verify correctness by comparing to the CPU computed maximum.
 *
 * 3. Data type:
 *    For simplicity we use integers. atomicMax is natively supported
 *    for integers on CUDA. (Using floats would require bitwise tricks.)
 *
 * 4. Implementation details:
 *    - The array size is configurable (default 1 << 24 ≈ 16M).
 *    - Block size of 256 threads is used.
 *    - The per-block array for the two-pass method is allocated in
 *      global memory; its size equals the number of blocks.
 *
 * 5. Output:
 *    The program prints:
 *      * CPU maximum
 *      * Single-kernel atomicMax result and time
 *      * Two-pass result and time
 *
 * 6. Edge cases:
 *    - The array size may not be divisible by blockSize*gridSize. We
 *      handle bounds checks inside kernels.
 *
 * This code is fully self-contained and can be compiled with nvcc:
 *   nvcc -O2 -arch=sm_70 max_reduction.cu -o max_reduction
 *********************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

// Error checking macro
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line "     \
                      << __LINE__ << ": " << cudaGetErrorString(err) << '\n';   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel that performs max reduction using atomicMax on a global variable
__global__ void atomicMaxKernel(const int *data, int n, int *globalMax) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple elements in strided fashion
    for (int i = idx; i < n; i += stride) {
        atomicMax(globalMax, data[i]);
    }
}

// First pass of the two-pass reduction: compute per-block max
__global__ void twoPassFirstKernel(const int *data, int n, int *blockMax) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize shared memory
    int localMax = INT_MIN;

    // Each thread processes multiple elements
    for (int i = idx; i < n; i += stride) {
        int val = data[i];
        if (val > localMax) localMax = val;
    }

    sdata[tid] = localMax;
    __syncthreads();

    // Reduce within block (binary tree)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // Write per-block max to global memory
    if (tid == 0) {
        blockMax[blockIdx.x] = sdata[0];
    }
}

// Second pass: reduce the per-block maxima
__global__ void twoPassSecondKernel(const int *blockMax, int numBlocks, int *globalMax) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Load block maxima into shared memory
    int localMax = INT_MIN;
    for (int i = idx; i < numBlocks; i += stride) {
        int val = blockMax[i];
        if (val > localMax) localMax = val;
    }
    sdata[tid] = localMax;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result from first block
    if (tid == 0) {
        atomicMax(globalMax, sdata[0]);
    }
}

int main() {
    // Parameters
    const size_t N = 1 << 24; // 16 million elements
    const int threadsPerBlock = 256;
    const int maxGridSize = 65535;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > maxGridSize) blocks = maxGridSize;

    // Allocate host memory
    std::vector<int> h_data(N);
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = std::rand();
    }

    // Compute CPU max for verification
    int cpuMax = *std::max_element(h_data.begin(), h_data.end());
    std::cout << "CPU max: " << cpuMax << std::endl;

    // Allocate device memory
    int *d_data = nullptr;
    int *d_globalMaxAtomic = nullptr;
    int *d_blockMax = nullptr;
    int *d_globalMaxTwoPass = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_globalMaxAtomic, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blockMax, blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_globalMaxTwoPass, sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize global max variables
    int initVal = INT_MIN;
    CUDA_CHECK(cudaMemcpy(d_globalMaxAtomic, &initVal, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_globalMaxTwoPass, &initVal, sizeof(int), cudaMemcpyHostToDevice));

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ====================== Single-kernel atomicMax ======================
    CUDA_CHECK(cudaEventRecord(start));
    atomicMaxKernel<<<blocks, threadsPerBlock>>>(d_data, N, d_globalMaxAtomic);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float timeAtomic;
    CUDA_CHECK(cudaEventElapsedTime(&timeAtomic, start, stop));
    int h_globalMaxAtomic = 0;
    CUDA_CHECK(cudaMemcpy(&h_globalMaxAtomic, d_globalMaxAtomic, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "AtomicMax result: " << h_globalMaxAtomic
              << "  Time: " << timeAtomic << " ms" << std::endl;

    // ====================== Two-pass reduction ======================
    // First pass
    int sharedMemSize = threadsPerBlock * sizeof(int);
    CUDA_CHECK(cudaEventRecord(start));
    twoPassFirstKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_data, N, d_blockMax);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float timeFirst;
    CUDA_CHECK(cudaEventElapsedTime(&timeFirst, start, stop));

    // Second pass
    // We'll launch a single block for simplicity; if blockMax size is large,
    // we could use multiple blocks but for this example it's fine.
    int secondBlocks = 1;
    CUDA_CHECK(cudaEventRecord(start));
    twoPassSecondKernel<<<secondBlocks, threadsPerBlock, sharedMemSize>>>(d_blockMax, blocks, d_globalMaxTwoPass);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float timeSecond;
    CUDA_CHECK(cudaEventElapsedTime(&timeSecond, start, stop));

    int h_globalMaxTwoPass = 0;
    CUDA_CHECK(cudaMemcpy(&h_globalMaxTwoPass, d_globalMaxTwoPass, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Two-pass result: " << h_globalMaxTwoPass
              << "  Time: " << timeFirst + timeSecond << " ms" << std::endl;

    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_globalMaxAtomic));
    CUDA_CHECK(cudaFree(d_blockMax));
    CUDA_CHECK(cudaFree(d_globalMaxTwoPass));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```