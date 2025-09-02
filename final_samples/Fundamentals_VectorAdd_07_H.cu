/*
Implement `output[i] = 1.0f / input[i]` (reciprocal).

My thinking:
- We need a CUDA kernel that takes input and output arrays and performs the reciprocal operation element‑wise.
- Each thread will compute one element: if threadIdx.x + blockIdx.x * blockDim.x < N.
- Use float for input and output, with the literal 1.0f.
- On the host side, allocate arrays, initialize them, copy to the device, launch the kernel, copy the result back, and verify.
- Basic error checking is performed after CUDA API calls.
- A small N is chosen for demonstration, and a few results are printed.
- The program is self‑contained and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// CUDA kernel that computes the reciprocal of each element.
__global__ void reciprocalKernel(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = 1.0f / input[idx];
    }
}

int main(void) {
    const int N = 16;                  // Number of elements
    const int size = N * sizeof(float);

    // Allocate host memory
    float *h_input  = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    if (h_input == NULL || h_output == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input array with sample values
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(i + 1);   // Avoid zero to prevent division by zero
    }

    // Allocate device memory
    float *d_input  = NULL;
    float *d_output = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_input,  size));
    CHECK_CUDA(cudaMalloc((void **)&d_output, size));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    reciprocalKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    CHECK_CUDA(cudaGetLastError());          // Check for launch errors
    CHECK_CUDA(cudaDeviceSynchronize());     // Ensure completion

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Input\tReciprocal\n");
    for (int i = 0; i < N; ++i) {
        printf("%f\t%f\n", h_input[i], h_output[i]);
    }

    // Clean up
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return EXIT_SUCCESS;
}
