/*
Aim: Explain the advantages of a grid-stride loop. (Kernel is independent of launch size, can handle any `N`, better hardware utilization if launch is small).

Thinking:
- The grid‑stride loop pattern allows a kernel to process an arbitrary number of elements (`N`) without having to adjust the launch configuration (block/ grid size). 
- Each thread calculates its global index `i` and then repeatedly adds the total number of elements processed per kernel launch (`blockDim.x * gridDim.x`) to `i`, effectively “striding” over the data set. 
- This keeps the kernel logic identical regardless of the size of the problem or the number of threads launched, making it robust and easy to maintain. 
- It also improves hardware utilization: if the grid launch is small (e.g., few blocks or threads), the loop ensures that all elements are still processed, preventing idle threads. 
- In practice, this means we can launch a kernel with a small, hardware‑friendly block size and still process large arrays efficiently, which is especially useful on GPUs where launching many blocks may be unnecessary or sub‑optimal. 
- The code below demonstrates a simple vector operation (scaling each element by 2.0) using a grid‑stride loop, shows independence from launch size, handles arbitrary `N`, and includes timing to illustrate performance with a modest launch configuration.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                          \
    {                                                              \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    #call, __FILE__, __LINE__,                    \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

__global__ void vectorScaleGridStride(float *d_out, const float *d_in, int N, float scale)
{
    // Calculate the starting index for this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid‑stride loop
    for (int i = idx; i < N; i += stride)
    {
        d_out[i] = d_in[i] * scale;
    }
}

int main(void)
{
    const int N = 1 << 24;          // ~16 million elements
    const int bytes = N * sizeof(float);
    const float scale = 2.0f;

    // Allocate host memory
    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        h_in[i] = (float)i;
    }

    // Allocate device memory
    float *d_in = NULL;
    float *d_out = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_out, bytes));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Choose a modest launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = 8;   // Small grid for demonstration

    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Launch kernel
    CHECK_CUDA(cudaEventRecord(start));
    vectorScaleGridStride<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, N, scale);
    CHECK_CUDA(cudaEventRecord(stop));

    // Wait for completion
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = h_in[i] * scale;
        if (fabs(h_out[i] - expected) > 1e-5) {
            errors++;
            printf("Mismatch at index %d: expected %f, got %f\n", i, expected, h_out[i]);
        }
    }
    if (errors == 0) {
        printf("Verification passed for first 10 elements.\n");
    } else {
        printf("Verification failed with %d mismatches.\n", errors);
    }

    // Clean up
    free(h_in);
    free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
