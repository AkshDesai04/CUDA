```cuda
/*
Aim of the program: Verify the results on the CPU.

Thought process:
- The goal is to demonstrate a simple CUDA kernel that performs a computation on the GPU
  (here, vector addition) and then to verify that the GPU-computed result matches
  the CPU-computed reference result.
- The program will:
  1. Allocate two input arrays and one output array on the host.
  2. Initialize the input arrays with deterministic values.
  3. Allocate corresponding device memory and copy inputs to the GPU.
  4. Launch a CUDA kernel that adds the two input vectors element-wise.
  5. Copy the GPU result back to host memory.
  6. Compute the reference result on the CPU.
  7. Compare the GPU and CPU results element by element, reporting any mismatches.
  8. Clean up all allocated memory.
- Error checking will be done after every CUDA API call and kernel launch.
- For simplicity, the vector size is set to a moderate value (e.g., 1<<20 elements)
  but it can be changed by defining VECTOR_SIZE.
- The kernel uses a simple 1D grid of thread blocks for clarity.
- Timing is optional but included to give an idea of GPU vs CPU performance.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define VECTOR_SIZE (1 << 20)   // 1,048,576 elements
#define THREADS_PER_BLOCK 256

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel to perform element-wise vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// CPU reference implementation for vector addition
void cpuVectorAdd(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    // Host allocations
    float *h_A = (float *)malloc(VECTOR_SIZE * sizeof(float));
    float *h_B = (float *)malloc(VECTOR_SIZE * sizeof(float));
    float *h_C_gpu = (float *)malloc(VECTOR_SIZE * sizeof(float));
    float *h_C_cpu = (float *)malloc(VECTOR_SIZE * sizeof(float));

    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize input arrays
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(VECTOR_SIZE - i);
    }

    // Device allocations
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, VECTOR_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, VECTOR_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, VECTOR_SIZE * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int blocks = (VECTOR_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 gridDim(blocks);
    dim3 blockDim(THREADS_PER_BLOCK);

    // Launch kernel and measure GPU time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    vectorAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, VECTOR_SIZE);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("GPU vector addition took %.3f ms.\n", milliseconds);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference computation
    clock_t cpu_start = clock();
    cpuVectorAdd(h_A, h_B, h_C_cpu, VECTOR_SIZE);
    clock_t cpu_end = clock();
    double cpu_time_ms = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU vector addition took %.3f ms.\n", cpu_time_ms);

    // Verify results
    int mismatches = 0;
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > 1e-5f) {
            if (mismatches < 10) {
                printf("Mismatch at index %d: GPU=%f, CPU=%f\n",
                       i, h_C_gpu[i], h_C_cpu[i]);
            }
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("Verification PASSED: GPU and CPU results match.\n");
    } else {
        printf("Verification FAILED: %d mismatches found.\n", mismatches);
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```