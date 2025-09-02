/*
Aim: Split a large vector addition task across two GPUs. Allocate half of the A and B vectors on GPU 0 and the other half on GPU 1. Launch a kernel on each GPU to process its half. The host must manage both devices, switching context with `cudaSetDevice`.

Thinking:
The program should demonstrate how to distribute a computational workload across multiple GPUs using CUDA. We need to:
1. Determine the number of available GPUs and ensure there are at least two.
2. Allocate a large input vector A and B on the host, and a result vector C.
3. Split the input vectors into two halves.
4. For each half:
   - Switch context to the appropriate GPU with cudaSetDevice.
   - Allocate device memory for that half of A, B, and C.
   - Copy the corresponding half of the host data to device memory.
   - Launch a simple element-wise addition kernel.
   - Copy the resulting half of C back to the host.
   - Free device memory for that half.
5. After both halves are processed, the host C vector contains the full result.
6. Optionally, verify a few entries to ensure correctness.
7. Wrap all CUDA API calls with error checking for robustness.

We will use a straightforward kernel that takes device pointers to A, B, and C and the number of elements to process. The kernel will compute C[i] = A[i] + B[i] for each i. The host code will orchestrate memory allocation, data transfer, kernel launch, and cleanup. The code is self-contained in a single .cu file.

Now the actual implementation follows.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Simple vector addition kernel
__global__ void vecAdd(const float *A, const float *B, float *C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    // Define total vector size
    const size_t N_total = 1 << 24; // ~16 million elements
    const size_t bytes_total = N_total * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes_total);
    float *h_B = (float *)malloc(bytes_total);
    float *h_C = (float *)malloc(bytes_total);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // Initialize input vectors
    for (size_t i = 0; i < N_total; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N_total - i);
    }

    // Query device count
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        fprintf(stderr, "At least two GPUs are required. Found %d.\n", deviceCount);
        free(h_A);
        free(h_B);
        free(h_C);
        exit(EXIT_FAILURE);
    }

    // Split workload
    size_t N_half = N_total / 2;
    size_t bytes_half = N_half * sizeof(float);

    // Process first half on GPU 0
    CHECK_CUDA(cudaSetDevice(0));

    float *d_A0 = NULL, *d_B0 = NULL, *d_C0 = NULL;
    CHECK_CUDA(cudaMalloc(&d_A0, bytes_half));
    CHECK_CUDA(cudaMalloc(&d_B0, bytes_half));
    CHECK_CUDA(cudaMalloc(&d_C0, bytes_half));

    CHECK_CUDA(cudaMemcpy(d_A0, h_A, bytes_half, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B0, h_B, bytes_half, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N_half + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A0, d_B0, d_C0, N_half);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C0, bytes_half, cudaMemcpyDeviceToHost));

    // Clean up GPU 0 memory
    CHECK_CUDA(cudaFree(d_A0));
    CHECK_CUDA(cudaFree(d_B0));
    CHECK_CUDA(cudaFree(d_C0));

    // Process second half on GPU 1
    CHECK_CUDA(cudaSetDevice(1));

    float *d_A1 = NULL, *d_B1 = NULL, *d_C1 = NULL;
    CHECK_CUDA(cudaMalloc(&d_A1, bytes_half));
    CHECK_CUDA(cudaMalloc(&d_B1, bytes_half));
    CHECK_CUDA(cudaMalloc(&d_C1, bytes_half));

    // Offsets into host arrays
    CHECK_CUDA(cudaMemcpy(d_A1, h_A + N_half, bytes_half, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B1, h_B + N_half, bytes_half, cudaMemcpyHostToDevice));

    // Launch kernel
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A1, d_B1, d_C1, N_half);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C + N_half, d_C1, bytes_half, cudaMemcpyDeviceToHost));

    // Clean up GPU 1 memory
    CHECK_CUDA(cudaFree(d_A1));
    CHECK_CUDA(cudaFree(d_B1));
    CHECK_CUDA(cudaFree(d_C1));

    // Optional: Verify a few elements
    bool success = true;
    for (size_t i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %zu: got %f, expected %f\n", i, h_C[i], expected);
            success = false;
            break;
        }
    }
    for (size_t i = N_total - 10; i < N_total; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %zu: got %f, expected %f\n", i, h_C[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition succeeded on both GPUs.\n");
    } else {
        printf("Vector addition failed.\n");
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset devices
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaDeviceReset());
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
