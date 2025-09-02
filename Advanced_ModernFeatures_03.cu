/*
 * Aim: Implement a simple program using Dynamic Parallelism. A parent kernel is launched from the host. Inside this parent kernel, based on some condition, it launches a child kernel. For example, the parent kernel iterates over a data structure, launching a child kernel to process each node.
 *
 * Thinking:
 * - Dynamic Parallelism requires compute capability >= 3.5. The parent kernel can launch child kernels from device.
 * - We'll create a simple data set: an array of integers on the device.
 * - Parent kernel will iterate over each element. If the element satisfies a condition (greater than a threshold), it will launch a child kernel to process that element; otherwise it writes the original value to output.
 * - Child kernel will just multiply the value by 2 and write to the output array at the same index.
 * - Because each parent thread may launch its own child kernel, this demonstrates dynamic parallelism.
 * - We need to compile with -rdc=true to enable relocatable device code and dynamic parallelism.
 * - Use a macro to check CUDA errors. Because kernel launches from device cannot directly return errors, we do not perform error checks inside child launches; host side error checking after device synchronization will capture any launch failures.
 * - Allocate and copy data to device, launch parent kernel, synchronize, copy back output, print results.
 * - Simple example ensures that child kernel launches are correct and that output reflects processing only on selected elements.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Child kernel: process a single element (multiply by 2)
__global__ void childKernel(int value, int *output, int idx) {
    // Since this is a simple demonstration, we launch with 1 thread
    output[idx] = value * 2;
}

// Parent kernel: iterates over the array and conditionally launches child kernels
__global__ void parentKernel(const int *input, int *output, int n, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = input[idx];
        if (val > threshold) {
            // Dynamically launch child kernel to process this element
            childKernel<<<1, 1>>>(val, output, idx);
        } else {
            // No child kernel needed; copy value unchanged
            output[idx] = val;
        }
    }
}

int main() {
    const int N = 10;
    const int threshold = 5;

    // Host arrays
    int h_input[N];
    int h_output[N];

    // Initialize host input array
    for (int i = 0; i < N; ++i) {
        h_input[i] = i;
    }

    // Device arrays
    int *d_input = nullptr;
    int *d_output = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_input, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, N * sizeof(int)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch parent kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    parentKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, threshold);

    // Synchronize to catch any launch errors
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Input : ");
    for (int i = 0; i < N; ++i) printf("%d ", h_input[i]);
    printf("\nOutput: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_output[i]);
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
