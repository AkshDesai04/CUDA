/*
Apply the same boundary check logic to a vector squaring kernel: `if (i < N) out[i] = in[i] * in[i]`.

Below is a complete CUDA program that demonstrates this logic. 
I chose to implement a simple end‑to‑end workflow:
1. Allocate a host vector of size N and fill it with sample values.
2. Allocate device memory for input and output vectors.
3. Copy the input vector from host to device.
4. Launch a CUDA kernel that squares each element, guarded by the boundary check `if (i < N)`.
5. Copy the result back to the host and print a few elements to verify correctness.
6. Clean up all allocated memory.

The kernel uses the classic `threadIdx.x + blockIdx.x * blockDim.x` calculation to determine the global index `i`. 
The boundary check prevents any thread from accessing out‑of‑bounds memory when the total number of threads exceeds N. 
The program also includes simple error checking for CUDA API calls to aid debugging.

This example is intentionally kept straightforward to focus on the boundary‑checking pattern requested in the prompt.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that squares each element with boundary check
__global__ void vector_square(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float val = in[i];
        out[i] = val * val;
    }
}

// Utility macro for CUDA error checking
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void) {
    const int N = 1 << 20; // 1M elements
    const int bytes = N * sizeof(float);

    // Host allocations
    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_in[i] = (float)i * 0.001f;
    }

    // Device allocations
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc((void**)&d_in,  bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out, bytes));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vector_square<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample results:\n");
    for (int i = 0; i < 5; ++i) {
        printf("h_in[%d] = %f, h_out[%d] = %f\n", i, h_in[i], i, h_out[i]);
    }

    // Clean up
    free(h_in);
    free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return EXIT_SUCCESS;
}
