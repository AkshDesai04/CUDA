```c
/* 
Implement `output[i] = 1.0f / input[i]` (reciprocal).

The goal of this program is to demonstrate a simple CUDA kernel that computes the reciprocal of each element in a float array. 
We allocate an array of floats on the host, initialize it with some sample values, copy it to the GPU, launch a kernel that
performs the reciprocal operation, and then copy the result back to the host to verify correctness. 
The kernel is launched with a grid of blocks and threads that cover the entire array. 
We include basic CUDA error checking for robustness and print the results to standard output.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA_ERROR(call)                                               \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",     \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call);\
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// CUDA kernel that computes the reciprocal of each element
__global__ void reciprocalKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Avoid division by zero; set output to 0.0f if input is zero
        if (input[idx] != 0.0f)
            output[idx] = 1.0f / input[idx];
        else
            output[idx] = 0.0f;
    }
}

int main(void) {
    const int N = 10;               // Number of elements
    const int SIZE = N * sizeof(float);

    // Allocate host memory
    float h_input[N];
    float h_output[N];

    // Initialize input array with sample values (non-zero to avoid division by zero)
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(i + 1);  // 1.0, 2.0, ..., 10.0
    }

    // Allocate device memory
    float *d_input = NULL;
    float *d_output = NULL;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, SIZE));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, SIZE));

    // Copy input data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, SIZE, cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    reciprocalKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, SIZE, cudaMemcpyDeviceToHost));

    // Print results
    printf("Input\t\tReciprocal\n");
    for (int i = 0; i < N; ++i) {
        printf("%f\t%f\n", h_input[i], h_output[i]);
    }

    // Clean up device memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    return 0;
}
```