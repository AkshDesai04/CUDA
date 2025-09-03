/*
Implement scalar subtraction: `output[i] = input[i] - scalar_value`.
The goal of this program is to demonstrate a simple CUDA kernel that
subtracts a scalar value from every element of an input array and stores
the result in an output array.  It includes:
- Host-side allocation and initialization of an input array.
- Allocation of device memory for both input and output.
- Copying data from host to device.
- Launching a CUDA kernel that performs element-wise subtraction.
- Copying the results back to the host.
- Basic error checking for CUDA API calls.
- Printing a small sample of results to verify correctness.

The kernel is intentionally straightforward: each thread handles one
array element.  We compute the global thread index and guard against
out-of-bounds indices.  This is a canonical example for learning
CUDA memory transfer and kernel execution patterns.

The code uses float precision, a fixed array size, and a simple
initialization pattern for clarity.  In real-world applications, one
might replace these with command-line arguments, dynamic sizing,
or more complex data structures.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

__global__ void scalarSubKernel(float *output, const float *input,
                                float scalar, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] - scalar;
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);
    const float scalar_value = 5.0f;

    // Allocate host memory
    float *h_input  = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host input array
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)i * 0.5f; // example data
    }

    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_input, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_output, bytes));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    scalarSubKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_input,
                                                        scalar_value, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 results
    printf("First 10 results of output[i] = input[i] - %f:\n", scalar_value);
    for (int i = 0; i < 10; ++i) {
        printf("output[%d] = %f\n", i, h_output[i]);
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return EXIT_SUCCESS;
}
