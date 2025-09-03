```cpp
/*
If you are using CUDA events, wrap `cudaEventCreate`, `cudaEventRecord`, and `cudaEventDestroy`.

Thinking process:
- The goal is to provide a set of wrapper functions around the core CUDA event APIs
  so that error handling, logging, or additional behavior can be added in a single place.
- Each wrapper should mirror the original API signature but add a check for the return
  status and report any errors via `cudaGetErrorString`. This keeps the rest of the
  application code cleaner and more readable.
- To demonstrate usage, the program will launch a simple kernel that initializes
  an array, and use the wrapped events to time the kernel execution.
- The main program flow:
  1. Allocate device memory for an array.
  2. Create two events using the wrappers (`start` and `stop`).
  3. Record the start event, launch the kernel, record the stop event.
  4. Synchronize and calculate elapsed time.
  5. Destroy the events and free device memory.
- The program compiles as a standard CUDA .cu file and can be built with `nvcc`.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

// Wrapper for cudaEventCreate with error checking
void myEventCreate(cudaEvent_t *event, unsigned int flags = 0) {
    cudaError_t err = cudaEventCreateWithFlags(event, flags);
    if (err != cudaSuccess) {
        std::cerr << "cudaEventCreateWithFlags failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Wrapper for cudaEventRecord with error checking
void myEventRecord(cudaEvent_t event, cudaStream_t stream = 0) {
    cudaError_t err = cudaEventRecord(event, stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaEventRecord failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Wrapper for cudaEventDestroy with error checking
void myEventDestroy(cudaEvent_t event) {
    cudaError_t err = cudaEventDestroy(event);
    if (err != cudaSuccess) {
        std::cerr << "cudaEventDestroy failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Simple kernel that fills an array with sequential values
__global__ void initKernel(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] = idx;
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(int);

    // Allocate device memory
    int *d_arr = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_arr, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // Create CUDA events using wrappers
    cudaEvent_t start, stop;
    myEventCreate(&start);
    myEventCreate(&stop);

    // Record start event
    myEventRecord(start);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    initKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Record stop event
    myEventRecord(stop);

    // Wait for the events to complete
    cudaDeviceSynchronize();

    // Calculate elapsed time
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms" << std::endl;

    // Clean up
    myEventDestroy(start);
    myEventDestroy(stop);
    cudaFree(d_arr);

    return 0;
}
```