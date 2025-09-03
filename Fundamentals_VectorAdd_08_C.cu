/*
Aim of the program:
Implement scalar division: `output[i] = input[i] / scalar_value`.

Thinking process:
1. The task is straightforward: given an array of floating point numbers `input`, divide each element by a scalar and store the result in `output`.
2. Since we need to use CUDA, the core of the program is a kernel that performs the division for each index `i`. The kernel will compute the global thread index and perform the division if the index is within bounds.
3. Host code responsibilities:
   - Allocate and initialize host data (an example input array).
   - Allocate device memory for input and output.
   - Copy input data from host to device.
   - Launch the kernel with a sufficient number of blocks/threads to cover the data size.
   - Copy the resulting output back to the host.
   - Verify or print some results.
   - Clean up resources.
4. For simplicity, we will use `float` as the data type and a fixed scalar value (e.g., 2.0f). The program will create a small array (e.g., 1024 elements) for demonstration, but the logic can scale to larger sizes.
5. Error checking macros will be added to handle CUDA API errors and kernel launch errors.
6. The program will be self‑contained and compile with `nvcc`. The output is written to the console, showing the first few input/output pairs.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

// Kernel: each thread handles one element
__global__ void scalar_division(const float* __restrict__ input,
                                float* __restrict__ output,
                                float scalar,
                                int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        output[idx] = input[idx] / scalar;
}

int main(void)
{
    const int N = 1024;                 // Number of elements
    const float scalar_value = 2.0f;    // Scalar to divide by

    // Host allocations
    float* h_input  = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input with some values (e.g., 0.0f, 1.0f, ..., N-1)
    for (int i = 0; i < N; ++i)
        h_input[i] = (float)i;

    // Device allocations
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    scalar_division<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, scalar_value, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 results
    printf("First 10 input/output values after division by %.2f:\n", scalar_value);
    for (int i = 0; i < 10; ++i)
        printf("input[%d] = %.2f -> output[%d] = %.2f\n", i, h_input[i], i, h_output[i]);

    // Clean up
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}
