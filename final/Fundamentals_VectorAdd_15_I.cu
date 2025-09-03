```cpp
/*
The `sinf` and `cosf` functions can sometimes be computed together more cheaply. 
Write a kernel that computes `C[i] = sinf(A[i])` and `D[i] = cosf(A[i])` and compare to separate kernels.

Thinking:
1. Allocate input array A and output arrays C and D on host. Fill A with random floats.
2. Allocate corresponding device arrays dA, dC, dD.
3. Implement three kernels:
   - sin_cos_kernel: computes both sinf and cosf in one pass.
   - sin_kernel: computes only sinf.
   - cos_kernel: computes only cosf.
4. Use cudaEvent_t to time each kernel execution.
5. Run the combined kernel once, then run sin_kernel followed by cos_kernel.
6. Copy back results and verify that the combined result matches the separate results within a small tolerance.
7. Print timing information for comparison.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define N (1 << 20)          // 1M elements
#define THREADS_PER_BLOCK 512

// Kernel that computes both sinf and cosf
__global__ void sin_cos_kernel(const float *A, float *C, float *D, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = A[idx];
        C[idx] = sinf(x);
        D[idx] = cosf(x);
    }
}

// Kernel that computes only sinf
__global__ void sin_kernel(const float *A, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = sinf(A[idx]);
    }
}

// Kernel that computes only cosf
__global__ void cos_kernel(const float *A, float *D, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        D[idx] = cosf(A[idx]);
    }
}

// Utility function to check CUDA errors
void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Host arrays
    float *hA = (float*)malloc(N * sizeof(float));
    float *hC = (float*)malloc(N * sizeof(float));
    float *hD = (float*)malloc(N * sizeof(float));
    float *hC_sep = (float*)malloc(N * sizeof(float));
    float *hD_sep = (float*)malloc(N * sizeof(float));

    // Seed random and fill A with random floats in range [-PI, PI]
    srand((unsigned)time(nullptr));
    for (int i = 0; i < N; ++i) {
        hA[i] = ((float)rand() / RAND_MAX) * 2.0f * M_PI - M_PI;
    }

    // Device arrays
    float *dA, *dC, *dD;
    cudaMalloc((void**)&dA, N * sizeof(float));
    cudaMalloc((void**)&dC, N * sizeof(float));
    cudaMalloc((void**)&dD, N * sizeof(float));
    checkCudaError("malloc device arrays");

    // Copy input to device
    cudaMemcpy(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("memcpy hA to dA");

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. Combined kernel timing
    cudaEventRecord(start);
    sin_cos_kernel<<<blocks, THREADS_PER_BLOCK>>>(dA, dC, dD, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkCudaError("combined kernel launch");

    float ms_combined = 0.0f;
    cudaEventElapsedTime(&ms_combined, start, stop);

    // 2. Separate kernels timing
    cudaEventRecord(start);
    sin_kernel<<<blocks, THREADS_PER_BLOCK>>>(dA, dC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkCudaError("sin kernel launch");
    float ms_sin = 0.0f;
    cudaEventElapsedTime(&ms_sin, start, stop);

    cudaEventRecord(start);
    cos_kernel<<<blocks, THREADS_PER_BLOCK>>>(dA, dD, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkCudaError("cos kernel launch");
    float ms_cos = 0.0f;
    cudaEventElapsedTime(&ms_cos, start, stop);

    float ms_separate = ms_sin + ms_cos;

    // Copy results back
    cudaMemcpy(hC, dC, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hD, dD, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hC_sep, dC, N * sizeof(float), cudaMemcpyDeviceToHost); // same pointer, but copy separately
    cudaMemcpy(hD_sep, dD, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify correctness
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(hC[i] - hC_sep[i]) > 1e-5f || fabs(hD[i] - hD_sep[i]) > 1e-5f) {
            ok = false;
            printf("Mismatch at %d: combined (%.6f, %.6f) separate (%.6f, %.6f)\n",
                   i, hC[i], hD[i], hC_sep[i], hD_sep[i]);
            break;
        }
    }
    if (ok) {
        printf("Results match within tolerance.\n");
    } else {
        printf("Results do NOT match!\n");
    }

    // Print timing results
    printf("Combined kernel time: %.3f ms\n", ms_combined);
    printf("Separate kernels time: sin=%.3f ms, cos=%.3f ms, total=%.3f ms\n",
           ms_sin, ms_cos, ms_separate);

    // Clean up
    free(hA);
    free(hC);
    free(hD);
    free(hC_sep);
    free(hD_sep);
    cudaFree(dA);
    cudaFree(dC);
    cudaFree(dD);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```