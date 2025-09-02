/*
Use other atomic functions like `atomicSub`, `atomicMin`, `atomicMax`, `atomicExch`.

In this program we demonstrate how to use a variety of atomic functions available in CUDA:
- `atomicSub` decrements a global counter atomically.
- `atomicMin` finds the minimum value in an array atomically.
- `atomicMax` finds the maximum value in an array atomically.
- `atomicExch` atomically exchanges a value at a specific location in an array.

The kernel launches perform simple operations on an array of 256 integers:
1. `subKernel` decrements a counter by 1 for each thread using `atomicSub`.
2. `minKernel` updates a global minimum value using `atomicMin`.
3. `maxKernel` updates a global maximum value using `atomicMax`.
4. `exchKernel` exchanges the value at index 0 of the array with a new value using `atomicExch`.

After kernel execution we copy back results to the host and print:
- The final counter value.
- The computed minimum and maximum from the array.
- The value at array index 0 to confirm the exchange.

Error checking is performed for CUDA API calls, and all resources are freed before exit.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <climits>

#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

const int N = 256;

// Kernel to atomically subtract 1 from a counter
__global__ void subKernel(int *counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicSub(counter, 1);
    }
}

// Kernel to atomically compute the minimum
__global__ void minKernel(const int *data, int *minVal, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicMin(minVal, data[idx]);
    }
}

// Kernel to atomically compute the maximum
__global__ void maxKernel(const int *data, int *maxVal, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicMax(maxVal, data[idx]);
    }
}

// Kernel to atomically exchange a value at a given index
__global__ void exchKernel(int *data, int idx, int newVal) {
    atomicExch(&data[idx], newVal);
}

int main() {
    // Host data
    int h_data[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = i * 2;  // simple pattern: 0,2,4,...
    }

    int h_counter = 100;
    int h_min = INT_MAX;
    int h_max = INT_MIN;

    // Device pointers
    int *d_data = nullptr;
    int *d_counter = nullptr;
    int *d_min = nullptr;
    int *d_max = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_data, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_counter, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_min, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_max, sizeof(int)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_min, &h_min, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_max, &h_max, sizeof(int), cudaMemcpyHostToDevice));

    // Launch parameters
    int threadsPerBlock = 128;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 1. AtomicSub kernel
    subKernel<<<blocks, threadsPerBlock>>>(d_counter, N);
    CHECK_CUDA(cudaGetLastError());

    // 2. AtomicMin kernel
    minKernel<<<blocks, threadsPerBlock>>>(d_data, d_min, N);
    CHECK_CUDA(cudaGetLastError());

    // 3. AtomicMax kernel
    maxKernel<<<blocks, threadsPerBlock>>>(d_data, d_max, N);
    CHECK_CUDA(cudaGetLastError());

    // 4. AtomicExch kernel (swap index 0 with 999)
    exchKernel<<<1, 1>>>(d_data, 0, 999);
    CHECK_CUDA(cudaGetLastError());

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Final counter (should be 100 - %d = %d): %d\n", N, 100 - N, h_counter);
    printf("Computed minimum value: %d\n", h_min);
    printf("Computed maximum value: %d\n", h_max);
    printf("Value at index 0 after exchange: %d (expected 999)\n", h_data[0]);

    // Clean up
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_counter));
    CHECK_CUDA(cudaFree(d_min));
    CHECK_CUDA(cudaFree(d_max));

    return 0;
}
