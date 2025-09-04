```cuda
/*
Generalize the program to split the work across all available GPUs on the system.
The goal is to demonstrate how a CUDA application can detect all GPUs present on
the system, partition a computational workload among them, and run each
partition concurrently.  In this example we perform a simple vector
addition (C = A + B) on a large array of floats.  The array is split into
chunks, one chunk per GPU.  Each GPU is assigned to a separate host thread
so that the kernels can execute in parallel on the hardware.  The key steps
are:

1. Detect the number of GPUs with cudaGetDeviceCount().
2. Allocate a single large host array for A, B and C (using pinned memory
   for faster transfer if desired).
3. Partition the data so that each GPU processes roughly N/GPU_COUNT
   elements, with any remainder handled by the last GPU.
4. For each GPU:
   - Set the current device with cudaSetDevice().
   - Allocate device memory for its chunk of A, B and C.
   - Copy the corresponding slice of the host arrays to the GPU.
   - Launch a simple element‑wise addition kernel.
   - Copy the result back to the host slice.
   - Clean up device memory.
5. Use std::thread to run each GPU’s work concurrently.
6. After all threads join, we can optionally verify the result.

This pattern can be generalized to any computation that can be partitioned
into independent chunks.  Care is taken to handle errors with a
checkCudaErrors macro.  The program is written in CUDA C++ and can be
compiled with nvcc.  No external libraries are required beyond the CUDA
runtime and the C++ standard library.
*/
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <vector>
#include <cassert>

// Simple CUDA error checking macro
#define checkCudaErrors(err) \
    do { \
        cudaError_t err__ = (err); \
        if (err__ != cudaSuccess) { \
            std::cerr << "CUDA error at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err__) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel performing element‑wise addition
__global__ void addKernel(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Function that each thread will execute for a specific GPU
void runGPU(int device_id,
            const float* hA, const float* hB, float* hC,
            int start_idx, int count) {
    // Set the current device
    checkCudaErrors(cudaSetDevice(device_id));

    // Allocate device memory
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    checkCudaErrors(cudaMalloc((void**)&dA, count * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&dB, count * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&dC, count * sizeof(float)));

    // Copy input slices to device
    checkCudaErrors(cudaMemcpy(dA, hA + start_idx, count * sizeof(float),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dB, hB + start_idx, count * sizeof(float),
                               cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, count);
    checkCudaErrors(cudaGetLastError());

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(hC + start_idx, dC, count * sizeof(float),
                               cudaMemcpyDeviceToHost));

    // Clean up
    checkCudaErrors(cudaFree(dA));
    checkCudaErrors(cudaFree(dB));
    checkCudaErrors(cudaFree(dC));
}

int main() {
    // Size of the full vectors
    const int N = 1 << 24; // 16 million elements

    // Determine how many GPUs are available
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable GPUs found." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Detected " << deviceCount << " GPU(s)." << std::endl;

    // Allocate host memory (using pageable memory for simplicity)
    std::vector<float> hA(N), hB(N), hC(N);

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(2 * i);
    }

    // Partition data among GPUs
    std::vector<int> startIdx(deviceCount);
    std::vector<int> count(deviceCount);
    int baseChunk = N / deviceCount;
    int remainder = N % deviceCount;
    int offset = 0;
    for (int i = 0; i < deviceCount; ++i) {
        startIdx[i] = offset;
        count[i] = baseChunk + (i < remainder ? 1 : 0);
        offset += count[i];
    }

    // Launch a thread for each GPU
    std::vector<std::thread> threads;
    threads.reserve(deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        threads.emplace_back(runGPU, i,
                             hA.data(), hB.data(), hC.data(),
                             startIdx[i], count[i]);
    }

    // Wait for all threads to finish
    for (auto& t : threads) {
        t.join();
    }

    // Optional verification
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (hC[i] != hA[i] + hB[i]) {
            std::cerr << "Mismatch at index " << i
                      << ": " << hC[i] << " != " << hA[i] + hB[i] << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) {
        std::cout << "Verification passed: all results are correct." << std::endl;
    } else {
        std::cout << "Verification failed." << std::endl;
    }

    return 0;
}
```