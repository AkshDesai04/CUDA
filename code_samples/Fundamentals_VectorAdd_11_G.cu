/*
Launch your grid-stride kernel with a number of blocks equal to the number of SMs on your GPU, and a reasonable block size (e.g. 128 or 256). This is a common performance heuristic.
In this implementation, I will demonstrate a simple kernel that squares each element of a large array using a grid-stride loop. The number of blocks is chosen to match the number of streaming multiprocessors (SMs) available on the device, which we query at runtime. I pick a block size of 256 threads, a typical value that balances occupancy and register usage. The kernel uses the standard grid-stride pattern:
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) { ... }
to allow each thread to work on multiple elements if the problem size exceeds the total number of threads in the grid. After launching the kernel, the host verifies a few results to ensure correctness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",             \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

__global__ void square_kernel(const float *in, float *out, size_t N)
{
    size_t stride = blockDim.x * gridDim.x;
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < N; i += stride)
    {
        float v = in[i];
        out[i] = v * v;
    }
}

int main(void)
{
    const size_t N = 1 << 24; // about 16 million elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_in  = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input
    for (size_t i = 0; i < N; ++i) {
        h_in[i] = (float)i;
    }

    // Allocate device memory
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_in,  bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Query device properties to get number of SMs
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int numSMs = prop.multiProcessorCount;
    int blockSize = 256; // Reasonable block size

    dim3 blockDim(blockSize);
    dim3 gridDim(numSMs);

    // Launch kernel
    square_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    bool ok = true;
    for (size_t i = 0; i < 10; ++i) {
        float expected = h_in[i] * h_in[i];
        if (h_out[i] != expected) {
            printf("Mismatch at index %zu: got %f, expected %f\n",
                   i, h_out[i], expected);
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("Verification passed for first 10 elements.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return EXIT_SUCCESS;
}
