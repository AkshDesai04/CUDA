```cuda
/*
 * Aim of the program (exactly as requested):
 * Compare the performance of the fused kernel (C=A+B, D=A-B) against two separate kernel launches (one for addition, one for subtraction). The fused kernel should be faster.
 *
 * Thinking process:
 * 1. I need to create a CUDA program that allocates two input arrays A and B, computes C = A + B and D = A - B, and measures the execution time.
 * 2. I will implement three kernels:
 *    - addKernel: computes C = A + B.
 *    - subKernel: computes D = A - B.
 *    - fusedKernel: computes both C and D in a single kernel launch.
 * 3. To compare performance, I will run the two separate kernels (add then sub) and time them with cudaEventRecord. Then I will run the fused kernel and time it similarly.
 * 4. I will use a large array size (e.g., 1 << 24 ≈ 16 million elements) to make kernel launch overhead negligible relative to compute time.
 * 5. I will allocate host memory, initialize it with random floats, copy to device, run the kernels, copy results back, and free memory.
 * 6. I will check for errors after each CUDA API call using a helper macro.
 * 7. After timing, I will print the elapsed times for both approaches and compute the speedup factor.
 * 8. I will also perform a simple correctness check on a few elements to ensure the fused kernel produces the same results as the separate kernels.
 * 9. The program will be written entirely in CUDA C, compiled with nvcc, and will not output any text besides the timing results.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel to compute C = A + B
__global__ void addKernel(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

// Kernel to compute D = A - B
__global__ void subKernel(const float *A, const float *B, float *D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) D[idx] = A[idx] - B[idx];
}

// Fused kernel to compute both C = A + B and D = A - B
__global__ void fusedKernel(const float *A, const float *B, float *C, float *D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = A[idx];
        float b = B[idx];
        C[idx] = a + b;
        D[idx] = a - b;
    }
}

int main() {
    const int N = 1 << 24; // 16 million elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_D = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C || !h_D) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with deterministic values
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.5f;
        h_B[i] = static_cast<float>(i) * 1.2f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_D, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ----------------- Separate Kernels -----------------
    CHECK_CUDA(cudaEventRecord(start));
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    subKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float msSeparate = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&msSeparate, start, stop));

    // Copy results back for verification
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // ----------------- Fused Kernel -----------------
    // Reuse d_C and d_D, no need to reset
    CHECK_CUDA(cudaEventRecord(start));
    fusedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float msFused = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&msFused, start, stop));

    // Copy fused results back
    float *h_C_fused = (float*)malloc(bytes);
    float *h_D_fused = (float*)malloc(bytes);
    CHECK_CUDA(cudaMemcpy(h_C_fused, d_C, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D_fused, d_D, bytes, cudaMemcpyDeviceToHost));

    // ----------------- Verification -----------------
    bool correct = true;
    for (int i = 0; i < N; i += N / 10) { // check 10 samples
        if (fabs(h_C[i] - h_C_fused[i]) > 1e-5f || fabs(h_D[i] - h_D_fused[i]) > 1e-5f) {
            correct = false;
            break;
        }
    }
    if (!correct) {
        fprintf(stderr, "Verification failed: results differ between separate and fused kernels.\n");
    }

    // ----------------- Output -----------------
    printf("Separate kernels elapsed time: %f ms\n", msSeparate);
    printf("Fused kernel elapsed time:    %f ms\n", msFused);
    if (msFused > 0.0f)
        printf("Speedup (Separate / Fused):   %f\n", msSeparate / msFused);
    else
        printf("Speedup: undefined (fused time zero)\n");

    // ----------------- Cleanup -----------------
    free(h_A); free(h_B); free(h_C); free(h_D);
    free(h_C_fused); free(h_D_fused);
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C)); CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```