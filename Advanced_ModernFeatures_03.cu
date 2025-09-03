```cuda
/*
Implement a simple program using Dynamic Parallelism. A parent kernel is launched from the host. Inside this parent kernel, based on some condition, it launches a child kernel. For example, the parent kernel iterates over a data structure, launching a child kernel to process each node.

Thinking:
- To demonstrate dynamic parallelism, the parent kernel will run on the device and launch a child kernel from within its body.
- We'll use a small integer array on the device. The parent kernel will iterate over the array indices.
- For each element that satisfies a condition (value > threshold), the parent kernel will launch a child kernel to modify that element (e.g., increment by 10).
- The child kernel is launched with a single thread and block since it's only processing one element. In a more complex scenario, the child could launch further kernels recursively.
- The host allocates and initializes the array, copies it to the device, launches the parent kernel, synchronizes, and copies the results back for printing.
- Error checking is added via a simple macro to make debugging easier.
- This program requires a GPU that supports dynamic parallelism (compute capability >= 3.5) and should be compiled with an appropriate CUDA compiler flag such as `-arch=sm_35` or higher.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define N 10  // Size of the array

// Simple macro for CUDA error checking
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Child kernel: increments the value at index idx by 10
__global__ void childKernel(int *data, int idx) {
    int val = data[idx];
    val += 10;
    data[idx] = val;
}

// Parent kernel: iterates over the array and launches child kernels conditionally
__global__ void parentKernel(int *data, int threshold) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        if (data[idx] > threshold) {
            // Launch child kernel to process this element
            childKernel<<<1, 1>>>(data, idx);
            // Optionally check for errors on child launch
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                // Since we are in device code, we cannot print, but we can write a sentinel
                // For simplicity, we ignore error handling here.
            }
        }
    }
}

int main() {
    // Host array initialization
    int h_data[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = i * 2;  // 0, 2, 4, ..., 18
    }

    // Allocate device memory
    int *d_data = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_data, N * sizeof(int)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    int threshold = 5;  // Elements greater than 5 will trigger child kernel

    // Launch parent kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    parentKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, threshold);

    // Wait for device to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Final array values:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaFree(d_data));
    return 0;
}
```