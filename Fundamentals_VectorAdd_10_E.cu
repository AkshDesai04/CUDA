```c
/* 
Aim of the program: For the GPU timing, use `cudaEvent`s around the kernel launch only. 
Thinking: 
- The program demonstrates how to time only the kernel execution on a CUDA GPU using cudaEvent_t. 
- It performs a simple vector addition of two arrays of floats. 
- Data transfer (host to device, device to host) is not included in the timing to isolate the kernel execution time. 
- We allocate host and device memory, initialize host vectors, copy them to the device, launch the kernel, record start and stop events around the kernel launch, synchronize, compute elapsed time, and print the result. 
- We also perform a simple verification of the result on the host. 
- The code includes basic error checking after CUDA API calls for robustness. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 20)  // Size of vectors (1 million elements)

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main(void) {
    // Host allocation
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Device allocation
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Timing using cudaEvent_t around the kernel launch only
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for kernel to finish
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print timing result
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Simple verification
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            if (errors < 10) { // print first few errors
                printf("Mismatch at index %d: host=%f, device=%f, expected=%f\n",
                       i, h_C[i], expected, expected);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("Verification: SUCCESS\n");
    } else {
        printf("Verification: FAILED with %d errors\n", errors);
    }

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
```