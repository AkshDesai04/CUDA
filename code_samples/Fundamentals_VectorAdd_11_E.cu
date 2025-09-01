```c
/*
Aim of the program: Implement a vector squaring kernel using a grid-stride loop.

Thinking:
1. The goal is to compute the square of each element in a vector on the GPU.
2. We will use a grid-stride loop so that the kernel can handle vectors larger than the total number of threads.
3. Steps:
   a. Allocate host memory for an input vector and an output vector.
   b. Allocate device memory for the same vectors.
   c. Copy input data from host to device.
   d. Launch the kernel with a reasonable block size (e.g., 256 threads per block).
      The kernel will iterate over its assigned indices using a grid-stride loop:
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < N;
             i += blockDim.x * gridDim.x) { ... }
   e. Each thread reads the input element, squares it, and writes the result to the output vector.
   f. Copy the result back to the host.
   g. Verify correctness by printing a few elements.
   h. Clean up memory.
4. Error checking: We'll wrap CUDA API calls with a helper macro to ensure any errors are caught early.
5. The code is self-contained, uses the CUDA runtime API, and should compile with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors following a CUDA API call or kernel launch */
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

/* Kernel: squares each element of the input vector and writes to the output vector */
__global__ void vectorSquareKernel(const float *input, float *output, int N) {
    /* Grid-stride loop to cover the entire vector */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        float val = input[i];
        output[i] = val * val;
    }
}

int main(void) {
    const int N = 1 << 20;          // Size of the vector (e.g., 1M elements)
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_input  = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    if (h_input == NULL || h_output == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input vector with some values */
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)i * 0.001f;   // Example data
    }

    /* Allocate device memory */
    float *d_input = NULL;
    float *d_output = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_input,  size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size));

    /* Copy input data to device */
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    /* Define block and grid dimensions */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    vectorSquareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError());   // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    /* Simple verification: print first 5 elements */
    printf("First 5 elements of the result vector:\n");
    for (int i = 0; i < 5; ++i) {
        printf("h_output[%d] = %f\n", i, h_output[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return EXIT_SUCCESS;
}
```