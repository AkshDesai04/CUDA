```cpp
/*
Aim: Allocate memory, launch a kernel that uses it, synchronize, and then free the memory.

Thought process:
- The program should demonstrate basic CUDA memory management: allocation on the device, usage by a kernel, synchronization, and cleanup.
- Allocate a device array using cudaMalloc.
- Write a simple kernel that populates each element with a deterministic value (e.g., its global index).
- Launch the kernel with a reasonable grid configuration (e.g., 128 threads per block and enough blocks to cover the array).
- Use cudaDeviceSynchronize() to ensure the kernel has finished before proceeding.
- Copy the results back to host memory with cudaMemcpy and print them to verify correctness.
- Add basic error checking for CUDA API calls and kernel launch failures.
- Finally, free the device memory with cudaFree and release any host resources.
- Keep the code minimal yet clear, suitable for compilation with nvcc.
*/

#include <iostream>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error in file " << __FILE__               \
                      << " at line " << __LINE__ << ": "                 \
                      << cudaGetErrorString(err) << std::endl;          \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Simple kernel that writes each element's global index into the array
__global__ void populateArray(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] = idx * 2; // Example: store double the index
    }
}

int main() {
    const int N = 1024; // Number of elements
    const int threadsPerBlock = 128;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory
    int *d_arr = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    // Launch kernel
    populateArray<<<blocks, threadsPerBlock>>>(d_arr, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Synchronize to ensure kernel completion
    CHECK_CUDA(cudaDeviceSynchronize());

    // Allocate host memory and copy results back
    int *h_arr = new int[N];
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print a few results for verification
    std::cout << "First 10 elements of the array:" << std::endl;
    for (int i = 0; i < 10 && i < N; ++i) {
        std::cout << "h_arr[" << i << "] = " << h_arr[i] << std::endl;
    }

    // Clean up
    delete[] h_arr;
    CHECK_CUDA(cudaFree(d_arr));

    return 0;
}
```