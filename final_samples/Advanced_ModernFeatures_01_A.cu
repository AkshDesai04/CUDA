```cuda
/*
Aim: Use `cudaMemAdvise` to give the driver hints about data usage patterns (e.g., set a vector to `cudaMemAdviseSetReadMostly` on the device).

Thinking:
- The goal is to demonstrate how to use cudaMemAdvise to inform the CUDA driver about expected memory access patterns.
- Allocate a large array on the host and copy it to device memory.
- Apply the cudaMemAdviseSetReadMostly hint to the device array to suggest that it will be read more frequently than written.
- Launch a simple kernel that reads the array many times to mimic a read‑heavy workload.
- Verify that the kernel executes correctly by checking a few output values.
- Clean up all allocated memory and exit.
- Error checking is added for clarity, but kept minimal for brevity.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Simple kernel that performs many reads from the input array
__global__ void readKernel(const float *in, float *out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float val = in[idx];
        float sum = 0.0f;
        for (int i = 0; i < 100; ++i)   // Simulate a read‑heavy operation
            sum += val;
        out[idx] = sum;
    }
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main(void)
{
    const int N = 1 << 20;           // 1M elements
    size_t size = N * sizeof(float);

    // Host allocation
    float *h_in  = (float *)malloc(size);
    float *h_out = (float *)malloc(size);
    if (!h_in || !h_out)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input
    for (int i = 0; i < N; ++i)
        h_in[i] = (float)i;

    // Device allocation
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_in, size));
    CUDA_CHECK(cudaMalloc((void **)&d_out, size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Give the driver a hint that d_in will be read mostly
    CUDA_CHECK(cudaMemAdvise(d_in, size, cudaMemAdviseSetReadMostly, 0));

    // Launch kernel
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    readKernel<<<blocks, threads>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Simple verification
    bool ok = true;
    for (int i = 0; i < 10; ++i)
    {
        float expected = h_in[i] * 100.0f;
        if (fabsf(h_out[i] - expected) > 1e-5f)
        {
            ok = false;
            printf("Mismatch at index %d: got %f, expected %f\n",
                   i, h_out[i], expected);
            break;
        }
    }
    if (ok)
        printf("Result verification passed.\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return 0;
}
```