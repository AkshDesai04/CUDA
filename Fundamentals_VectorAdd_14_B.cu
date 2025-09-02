```c
/*
Aim of the program:
Time the `double` precision kernel and compare it to the `float` version. On most consumer GPUs, double precision performance is significantly lower.

Thinking:
- We need a self‑contained CUDA C program that can be compiled with `nvcc` and run on a consumer GPU.
- The program will create two input vectors, `A` and `B`, and compute `C = A + B` using two separate kernels:
    1. `vecAddFloat` which uses `float` types.
    2. `vecAddDouble` which uses `double` types.
- The sizes of the vectors should be large enough to be measurable but not too large to cause out‑of‑memory errors; `1 << 20` (1,048,576) elements is a good compromise.
- We'll time each kernel using CUDA events (`cudaEvent_t`) because they provide high‑resolution GPU timestamps that are accurate for device code execution.
- After running each kernel, we'll copy the results back to host memory to ensure the kernel actually finished (the copy also forces synchronization but that's fine for timing here).
- We'll print the elapsed times in milliseconds for each precision and compute the ratio `float_time / double_time` to illustrate the performance gap.
- The program will also perform simple sanity checks on the results to ensure correctness.
- No external libraries or helpers are needed; all memory management is done with `cudaMalloc`/`cudaFree` and `cudaMemcpy`.
- Error handling will be minimal but sufficient: we check the return status of each CUDA API call and abort if something fails.

Note: Compile with:
    nvcc -O2 -arch=sm_35 -o vec_add_compare vec_add_compare.cu
(Replace `sm_35` with your GPU's compute capability if different.)
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define N (1 << 20)          // 1M elements
#define THREADS_PER_BLOCK 256

// Kernel for float addition
__global__ void vecAddFloat(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

// Kernel for double addition
__global__ void vecAddDouble(const double *A, const double *B, double *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

// Utility to check CUDA errors
inline void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    // Host allocations
    float *h_Af = (float*)malloc(N * sizeof(float));
    float *h_Bf = (float*)malloc(N * sizeof(float));
    float *h_Cf = (float*)malloc(N * sizeof(float));

    double *h_Ad = (double*)malloc(N * sizeof(double));
    double *h_Bd = (double*)malloc(N * sizeof(double));
    double *h_Cd = (double*)malloc(N * sizeof(double));

    // Initialize data
    for (int i = 0; i < N; ++i)
    {
        h_Af[i] = (float)i * 0.5f;
        h_Bf[i] = (float)(N - i) * 0.3f;

        h_Ad[i] = (double)i * 0.5;
        h_Bd[i] = (double)(N - i) * 0.3;
    }

    // Device allocations
    float *d_Af, *d_Bf, *d_Cf;
    double *d_Ad, *d_Bd, *d_Cd;

    checkCudaError(cudaMalloc((void**)&d_Af, N * sizeof(float)), "cudaMalloc d_Af");
    checkCudaError(cudaMalloc((void**)&d_Bf, N * sizeof(float)), "cudaMalloc d_Bf");
    checkCudaError(cudaMalloc((void**)&d_Cf, N * sizeof(float)), "cudaMalloc d_Cf");

    checkCudaError(cudaMalloc((void**)&d_Ad, N * sizeof(double)), "cudaMalloc d_Ad");
    checkCudaError(cudaMalloc((void**)&d_Bd, N * sizeof(double)), "cudaMalloc d_Bd");
    checkCudaError(cudaMalloc((void**)&d_Cd, N * sizeof(double)), "cudaMalloc d_Cd");

    // Copy inputs to device
    checkCudaError(cudaMemcpy(d_Af, h_Af, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy to d_Af");
    checkCudaError(cudaMemcpy(d_Bf, h_Bf, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy to d_Bf");
    checkCudaError(cudaMemcpy(d_Ad, h_Ad, N * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy to d_Ad");
    checkCudaError(cudaMemcpy(d_Bd, h_Bd, N * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy to d_Bd");

    // Launch parameters
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Events for timing
    cudaEvent_t start, stop;
    float elapsedTimeFloat = 0.0f;
    float elapsedTimeDouble = 0.0f;

    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop),  "cudaEventCreate stop");

    // ----------------- Float kernel timing -----------------
    checkCudaError(cudaEventRecord(start, 0), "cudaEventRecord start float");
    vecAddFloat<<<blocks, THREADS_PER_BLOCK>>>(d_Af, d_Bf, d_Cf, N);
    checkCudaError(cudaEventRecord(stop, 0), "cudaEventRecord stop float");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop float");
    checkCudaError(cudaEventElapsedTime(&elapsedTimeFloat, start, stop), "cudaEventElapsedTime float");

    // Copy result back
    checkCudaError(cudaMemcpy(h_Cf, d_Cf, N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy from d_Cf");

    // ----------------- Double kernel timing -----------------
    checkCudaError(cudaEventRecord(start, 0), "cudaEventRecord start double");
    vecAddDouble<<<blocks, THREADS_PER_BLOCK>>>(d_Ad, d_Bd, d_Cd, N);
    checkCudaError(cudaEventRecord(stop, 0), "cudaEventRecord stop double");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop double");
    checkCudaError(cudaEventElapsedTime(&elapsedTimeDouble, start, stop), "cudaEventElapsedTime double");

    // Copy result back
    checkCudaError(cudaMemcpy(h_Cd, d_Cd, N * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy from d_Cd");

    // ----------------- Output timings -----------------
    printf("Float kernel time:  %f ms\n", elapsedTimeFloat);
    printf("Double kernel time: %f ms\n", elapsedTimeDouble);
    if (elapsedTimeDouble > 0.0f)
        printf("Speedup factor (float/double): %f\n", elapsedTimeFloat / elapsedTimeDouble);
    else
        printf("Double kernel time too small to compute speedup.\n");

    // Simple correctness check (optional)
    int correct = 1;
    for (int i = 0; i < N; ++i)
    {
        float expected = h_Af[i] + h_Bf[i];
        if (fabs(h_Cf[i] - expected) > 1e-5f)
        {
            printf("Float result mismatch at index %d: got %f, expected %f\n", i, h_Cf[i], expected);
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("Float kernel produced correct results.\n");

    correct = 1;
    for (int i = 0; i < N; ++i)
    {
        double expected = h_Ad[i] + h_Bd[i];
        if (fabs(h_Cd[i] - expected) > 1e-12)
        {
            printf("Double result mismatch at index %d: got %lf, expected %lf\n", i, h_Cd[i], expected);
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("Double kernel produced correct results.\n");

    // Clean up
    cudaFree(d_Af); cudaFree(d_Bf); cudaFree(d_Cf);
    cudaFree(d_Ad); cudaFree(d_Bd); cudaFree(d_Cd);
    free(h_Af); free(h_Bf); free(h_Cf);
    free(h_Ad); free(h_Bd); free(h_Cd);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```