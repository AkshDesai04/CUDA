/*
Initialize the bin array to all zeros using `cudaMemset` before launching the kernel.

Thinking process:
1. The goal is to demonstrate how to use `cudaMemset` to initialize a device array to zero before launching a kernel.
2. The program will allocate an integer array (`d_bin`) on the GPU, set it to zero using `cudaMemset`, and then launch a simple kernel that writes a known value into each element.
3. After the kernel execution, the program copies the array back to the host and prints it to verify that the initialization succeeded.
4. For simplicity, the kernel will simply set each element to its index value (or any deterministic value) to show that the array was properly initialized.
5. Error checking is added for all CUDA API calls to ensure robustness.
6. The program is fully self‑contained and can be compiled with `nvcc` to produce an executable.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Simple kernel that writes a deterministic value to each bin
__global__ void initKernel(int *d_bin, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bins) {
        // For demonstration, set each bin to its index value
        d_bin[idx] = idx;
    }
}

int main(void) {
    const int num_bins = 10;
    int *d_bin = NULL;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_bin, num_bins * sizeof(int)));

    // Initialize the bin array to all zeros using cudaMemset
    CHECK_CUDA(cudaMemset(d_bin, 0, num_bins * sizeof(int)));

    // Launch kernel with enough threads to cover all bins
    const int threadsPerBlock = 256;
    const int blocks = (num_bins + threadsPerBlock - 1) / threadsPerBlock;
    initKernel<<<blocks, threadsPerBlock>>>(d_bin, num_bins);
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch error

    // Copy results back to host
    int h_bin[num_bins];
    CHECK_CUDA(cudaMemcpy(h_bin, d_bin, num_bins * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the contents of the bin array
    printf("Contents of bin array after kernel execution:\n");
    for (int i = 0; i < num_bins; ++i) {
        printf("bin[%d] = %d\n", i, h_bin[i]);
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_bin));

    return 0;
}
