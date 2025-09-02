```cuda
/*
Implement a kernel that computes the square root of every element: `output[i] = sqrtf(input[i])`.

Thought process:
1. The task is straightforward: apply sqrtf element‑wise to a float array.
2. We need a CUDA kernel that receives pointers to input and output arrays along with the array size.
3. Each thread will handle one element; we must guard against out‑of‑bounds indices.
4. In main() we allocate host memory, initialize an example input array, copy it to device,
   launch the kernel with a suitable grid/block configuration, copy the result back, and
   verify it by printing a few values.
5. Include error checking for CUDA API calls.
6. Use sqrtf (single‑precision) as specified.
*/

#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    __FUNCTION__, __FILE__, __LINE__,                  \
                    cudaGetErrorString(err));                         \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel: compute sqrtf for each element
__global__ void sqrt_kernel(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sqrtf(input[idx]);
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Host allocation
    float *h_input  = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // Initialize input with some values (e.g., 0..N-1)
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)i;
    }

    // Device allocation
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input,  size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    sqrt_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError()); // check launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 results
    printf("First 10 results (sqrt of input):\n");
    for (int i = 0; i < 10; ++i) {
        printf("sqrtf(%f) = %f\n", h_input[i], h_output[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}
```