/*
Allocate memory, launch a kernel that uses it, synchronize, and then free the memory.
The goal of this program is to demonstrate the basic CUDA memory workflow:
1. Allocate host memory.
2. Allocate device memory.
3. Copy data from host to device.
4. Launch a simple kernel that modifies the data on the device.
5. Synchronize to ensure kernel completion.
6. Copy the result back to the host.
7. Free both device and host memory.

My thinking when writing this:
- I chose a straightforward element‑wise operation (increment each array element) to keep the kernel simple.
- I used the CUDA runtime API (`cudaMalloc`, `cudaMemcpy`, etc.) for brevity and clarity.
- Basic error checking is added via a macro to keep the code clean.
- After kernel execution I synchronize with `cudaDeviceSynchronize` to guarantee that the device has finished before copying results back.
- Finally, I free both host and device resources to avoid memory leaks.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                     \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

__global__ void incrementKernel(int *d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] += 1;
    }
}

int main(void) {
    const int N = 1 << 20;           // 1M elements
    const size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_data = (int *)malloc(size);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    // Allocate device memory
    int *d_data = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_data, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Define kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Synchronize to ensure kernel completion
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample results after kernel execution:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_data[%d] = %d\n", i, h_data[i]);
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_data));

    // Free host memory
    free(h_data);

    // Reset device (optional)
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
