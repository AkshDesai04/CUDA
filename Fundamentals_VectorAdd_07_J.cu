/*
Aim of the program: Time the squaring kernel.

Thinking:
- The program will allocate a large array on the host, initialize it with some values,
  and then copy it to the device.
- A simple CUDA kernel will be launched that squares each element of the array.
- CUDA events will be used to time the duration of the kernel execution
  (from kernel launch to completion).
- After the kernel finishes, the result is copied back to the host
  (optional correctness check) and the elapsed time is printed.
- The program includes basic error checking and uses a configurable
  array size and block size.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that squares each element
__global__ void squareKernel(float *d_out, const float *d_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = d_in[idx];
        d_out[idx] = val * val;
    }
}

int main(void) {
    const int N = 1 << 26;  // ~67 million elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        h_in[i] = (float)i * 0.001f;  // arbitrary values
    }

    // Allocate device memory
    float *d_in = NULL;
    float *d_out = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_in,  bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Define execution configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, N);

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Optional: Verify a few results
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_in[i] * h_in[i];
        if (fabs(h_out[i] - expected) > 1e-5f) {
            ok = false;
            break;
        }
    }
    if (!ok) {
        fprintf(stderr, "Result verification failed!\n");
    } else {
        printf("Result verification succeeded.\n");
    }

    // Output timing information
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return 0;
}
