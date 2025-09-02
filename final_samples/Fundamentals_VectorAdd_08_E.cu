/*
Aim: Implement `output[i] = scalar_value - input[i]`.

Thinking:
1. Understand requirement: For each element in input array, compute scalar_value minus that element and store in output array.
2. Decide on data type: use float for simplicity. Scalar_value could be a constant or passed as argument.
3. Implement CUDA kernel that takes device pointers to input and output, scalar_value, and number of elements. Kernel uses thread index to compute element‑wise operation.
4. In host code, allocate host arrays, initialize with sample values.
5. Allocate device memory, copy input to device.
6. Launch kernel with appropriate block and grid sizes.
7. Copy output back to host and optionally print.
8. Add error checking after each CUDA API call.
9. Clean up device memory.
10. Ensure program compiles as a .cu file and runs with nvcc.

Edge cases:
- Ensure index does not exceed array length.
- For small N, compute block size accordingly.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10                 // Size of the arrays
#define BLOCK_SIZE 256       // Threads per block

// CUDA kernel to compute output[i] = scalar_value - input[i]
__global__ void subtract_scalar_kernel(const float *input, float *output, float scalar_value, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = scalar_value - input[idx];
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main(void)
{
    float h_input[N];
    float h_output[N];
    float scalar_value = 5.0f;  // Example scalar value

    // Initialize input array with some values
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)i;  // e.g., 0.0, 1.0, 2.0, ...
    }

    // Device pointers
    float *d_input = NULL;
    float *d_output = NULL;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_output, N * sizeof(float)));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Compute grid size
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    subtract_scalar_kernel<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, scalar_value, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    printf("scalar_value = %f\n", scalar_value);
    printf("input array:\n");
    for (int i = 0; i < N; ++i) {
        printf("%f ", h_input[i]);
    }
    printf("\noutput array (scalar_value - input):\n");
    for (int i = 0; i < N; ++i) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    // Free device memory
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
