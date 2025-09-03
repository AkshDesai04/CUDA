/*
Calculate the size in bytes required for the allocation (`256 * sizeof(int)`) and use that variable in both `cudaMalloc` and `cudaMemcpy`.

I first considered how to demonstrate the requirement to use a size variable in both CUDA API calls. The key is to compute the allocation size once, store it in a `size_t` variable, and then reuse that variable in `cudaMalloc` and in every `cudaMemcpy` call that deals with that allocation. To make the example meaningful I allocate an array of 256 integers on the host, copy it to the device, run a simple kernel that increments each element, copy it back, and print the first few results. I also add a small macro for error checking so that any CUDA runtime errors are reported cleanly. The code is self‑contained and ready to be compiled with `nvcc`. */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Simple kernel that adds 1 to each element
__global__ void add_one(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}

int main(void) {
    const int N = 256;
    size_t size = N * sizeof(int);      // <-- compute size once

    // Allocate host memory
    int h_data[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;                  // initialize with some values
    }

    // Allocate device memory using the size variable
    int *d_data = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_data, size));

    // Copy host to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_one<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError()); // check kernel launch

    // Copy device back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Print first 10 results to verify
    printf("First 10 results after increment:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
