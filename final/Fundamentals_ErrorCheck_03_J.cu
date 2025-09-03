```cpp
/* 
Refactor your vector add program to include post-launch error checking.
Think: The user wants a CUDA vector addition example that includes error checking after launching the kernel. We should start with a clear comment block at the top that states the aim verbatim, then provide our reasoning. We need to allocate host and device memory, copy input data, launch the kernel, then immediately check for launch errors with cudaGetLastError(), and also check for runtime errors after cudaDeviceSynchronize(). If any errors occur, print a message and exit. The kernel itself will perform element-wise addition. The host code will also verify the result or simply print a few elements to confirm. We should keep the code minimal but functional, using standard CUDA headers, error-checking macros, and proper cleanup. No extra output or text should be produced beyond the .cu file content. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel for element-wise vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int main(void) {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Post-launch error checking
    cudaError_t err = cudaGetLastError(); // Check for launch errors
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Wait for kernel to finish and check for runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA runtime error after kernel execution: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    bool success = true;
    for (int i = 0; i < 5; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Verification failed at index %d: %f != %f\n", i, h_C[i], expected);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition successful for first 5 elements.\n");
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```