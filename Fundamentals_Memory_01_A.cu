/*
After copying, use `cudaFree` to release the device memory.
[Thinking]
The goal of this CUDA program is to demonstrate the complete workflow of allocating memory on the GPU, copying data from host to device, performing a simple computation (incrementing each element), copying the result back to the host, and finally freeing the GPU memory with cudaFree. 
To illustrate this, the program:
1. Allocates an array on the host and initializes it.
2. Allocates corresponding memory on the device using cudaMalloc.
3. Copies the host data to the device with cudaMemcpy.
4. Launches a kernel that adds 1 to each element.
5. Copies the processed data back to the host with another cudaMemcpy.
6. Uses cudaFree to deallocate the device memory.
7. Includes basic error checking after each CUDA call.
The program uses a simple kernel `addOneKernel` which adds 1 to each element in the array. 
After execution, the host array will contain values incremented by 1, and the device memory will be released properly. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Simple kernel that adds 1 to each element
__global__ void addOneKernel(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

int main() {
    const int N = 1024;          // Number of elements
    const size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_arr = (int*)malloc(size);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Allocate device memory
    int *d_arr = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice));

    // Launch kernel: 256 threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addOneKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost));

    // Verify result
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != i + 1) {
            ++errors;
            if (errors <= 10) { // Print first few errors
                printf("Mismatch at index %d: expected %d, got %d\n",
                       i, i + 1, h_arr[i]);
            }
        }
    }
    if (errors == 0) {
        printf("All values incremented correctly.\n");
    } else {
        printf("Total mismatches: %d\n", errors);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_arr));

    // Free host memory
    free(h_arr);

    // Reset device (optional)
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
