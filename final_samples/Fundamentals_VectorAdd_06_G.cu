/*
Aim of the program: Verify the results of SAXPY on the CPU.

Thinking process:
1. Understand SAXPY: y = a * x + y, where a is scalar, x and y are vectors.
2. Goal: compute SAXPY on GPU, copy result back, compute SAXPY on CPU, compare the two results for correctness.
3. Steps:
   - Define vector size (e.g., 1 million elements) and scalar a.
   - Allocate host vectors x and y, initialize them with deterministic values for reproducibility.
   - Allocate device vectors x_d and y_d, copy host data to device.
   - Launch CUDA kernel to perform SAXPY on device.
   - Copy result y_d back to host (y_gpu).
   - Compute SAXPY on CPU, store in y_cpu.
   - Compare y_gpu and y_cpu element-wise within a small epsilon tolerance.
   - Print success or failure message.
4. Implementation details:
   - Use float for simplicity.
   - Implement error checking for CUDA API calls.
   - Use simple indexing in kernel (threadIdx + blockIdx * blockDim).
   - Choose block size (e.g., 256) and compute grid size accordingly.
   - Use standard C headers and CUDA runtime API.
   - Keep the program self‑contained in a single .cu file.
   - Avoid any external dependencies beyond the CUDA toolkit.
   - Add comments for clarity.
5. Edge cases: vector size may not be multiple of block size; kernel loop handles that.
6. Precision: Use epsilon = 1e-5 for float comparison.

The final code below follows this plan and can be compiled with nvcc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1000000           // Number of elements
#define BLOCK_SIZE 256      // Threads per block
#define EPSILON 1e-5f       // Tolerance for floating point comparison

// CUDA error checking macro
#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// SAXPY kernel: y = a * x + y
__global__ void saxpy_kernel(int n, float a, const float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

// Host function to perform SAXPY on CPU
void saxpy_cpu(int n, float a, const float *x, float *y) {
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

// Function to compare two arrays
int compare_arrays(const float *a, const float *b, int n, float eps) {
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > eps) {
            fprintf(stderr, "Mismatch at index %d: GPU=%f, CPU=%f, diff=%f\n",
                    i, a[i], b[i], diff);
            return 0; // failure
        }
    }
    return 1; // success
}

int main(void) {
    // Scalar
    float a = 2.5f;

    // Host memory allocation
    float *h_x = (float *)malloc(N * sizeof(float));
    float *h_y_gpu = (float *)malloc(N * sizeof(float));
    float *h_y_cpu = (float *)malloc(N * sizeof(float));

    if (!h_x || !h_y_gpu || !h_y_cpu) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize vectors
    for (int i = 0; i < N; ++i) {
        h_x[i] = (float)i / N;           // Example: ascending values between 0 and 1
        h_y_gpu[i] = 1.0f;               // y initialized to 1
        h_y_cpu[i] = h_y_gpu[i];         // Copy initial y for CPU computation
    }

    // Device memory allocation
    float *d_x = NULL;
    float *d_y = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_y, N * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y_gpu, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    saxpy_kernel<<<grid_size, BLOCK_SIZE>>>(N, a, d_x, d_y);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU SAXPY
    saxpy_cpu(N, a, h_x, h_y_cpu);

    // Compare results
    int success = compare_arrays(h_y_gpu, h_y_cpu, N, EPSILON);
    if (success) {
        printf("SAXPY verification succeeded: GPU and CPU results match within tolerance.\n");
    } else {
        printf("SAXPY verification failed: GPU and CPU results differ.\n");
    }

    // Clean up
    free(h_x);
    free(h_y_gpu);
    free(h_y_cpu);
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
