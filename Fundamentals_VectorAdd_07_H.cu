/*
Aim of the program:
Implement `output[i] = 1.0f / input[i]` (reciprocal).

Thinking process:
- The task is to compute the reciprocal of each element in an input array and store the result in an output array.
- We need a CUDA kernel that performs the operation element-wise.
- Host code must allocate memory on both host and device, transfer data, launch the kernel, and retrieve the results.
- Basic CUDA error checking will be added to ensure robustness.
- For demonstration, a small array of floats is initialized on the host, the kernel is invoked, and the results are printed.
- The kernel uses a simple index calculation based on blockIdx, blockDim, and threadIdx, with bounds checking.
- The code uses standard CUDA headers and follows the C style as requested.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err)                                     \
    if ((err) != cudaSuccess) {                             \
        fprintf(stderr, "CUDA error: %s\n",                 \
                cudaGetErrorString(err));                   \
        exit(EXIT_FAILURE);                                 \
    }

__global__ void reciprocal_kernel(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = 1.0f / input[idx];
    }
}

int main(void) {
    const int N = 10;                      // Number of elements
    size_t size = N * sizeof(float);

    // Host memory allocation
    float *h_input  = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    if (!h_input || !h_output) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(i + 1);   // 1.0, 2.0, ..., N
    }

    // Device memory allocation
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void **)&d_input, size));
    CUDA_CHECK(cudaMalloc((void **)&d_output, size));

    // Copy input from host to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    reciprocal_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Input\tReciprocal\n");
    for (int i = 0; i < N; ++i) {
        printf("%f\t%f\n", h_input[i], h_output[i]);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}
