/*
Time the `double` precision kernel and compare it to the `float` version. On most consumer GPUs, double precision performance is significantly lower.

My approach is to write a simple CUDA program that performs element-wise addition of two vectors, once using single-precision floats and once using double-precision doubles. The kernels are identical except for the data type. I will time each kernel execution using CUDA events to measure the GPU runtime accurately. The program allocates host and device memory, initializes data, copies it to the device, runs each kernel, records timings, and then prints the elapsed times. I will also include basic error checking and use a reasonably large vector size (e.g., 10 million elements) to expose the performance difference. The code is self-contained in a single .cu file, follows CUDA C conventions, and is ready to compile with nvcc.*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Simple vector addition kernel for float
__global__ void vecAddFloat(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// Simple vector addition kernel for double
__global__ void vecAddDouble(const double *a, const double *b, double *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    const int N = 10 * 1000 * 1000; // 10 million elements
    const size_t sizeFloat = N * sizeof(float);
    const size_t sizeDouble = N * sizeof(double);

    // Host allocations
    float *h_a_f = (float*)malloc(sizeFloat);
    float *h_b_f = (float*)malloc(sizeFloat);
    float *h_c_f = (float*)malloc(sizeFloat);

    double *h_a_d = (double*)malloc(sizeDouble);
    double *h_b_d = (double*)malloc(sizeDouble);
    double *h_c_d = (double*)malloc(sizeDouble);

    if (!h_a_f || !h_b_f || !h_c_f || !h_a_d || !h_b_d || !h_c_d) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_a_f[i] = sinf((float)i);
        h_b_f[i] = cosf((float)i);
        h_a_d[i] = sin((double)i);
        h_b_d[i] = cos((double)i);
    }

    // Device allocations
    float *d_a_f, *d_b_f, *d_c_f;
    double *d_a_d, *d_b_d, *d_c_d;

    CUDA_CHECK(cudaMalloc((void**)&d_a_f, sizeFloat));
    CUDA_CHECK(cudaMalloc((void**)&d_b_f, sizeFloat));
    CUDA_CHECK(cudaMalloc((void**)&d_c_f, sizeFloat));

    CUDA_CHECK(cudaMalloc((void**)&d_a_d, sizeDouble));
    CUDA_CHECK(cudaMalloc((void**)&d_b_d, sizeDouble));
    CUDA_CHECK(cudaMalloc((void**)&d_c_d, sizeDouble));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a_f, h_a_f, sizeFloat, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_f, h_b_f, sizeFloat, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_a_d, h_a_d, sizeDouble, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_d, h_b_d, sizeDouble, cudaMemcpyHostToDevice));

    // Launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Time float kernel
    CUDA_CHECK(cudaEventRecord(start, 0));
    vecAddFloat<<<blocksPerGrid, threadsPerBlock>>>(d_a_f, d_b_f, d_c_f, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msFloat = 0;
    CUDA_CHECK(cudaEventElapsedTime(&msFloat, start, stop));

    // Time double kernel
    CUDA_CHECK(cudaEventRecord(start, 0));
    vecAddDouble<<<blocksPerGrid, threadsPerBlock>>>(d_a_d, d_b_d, d_c_d, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msDouble = 0;
    CUDA_CHECK(cudaEventElapsedTime(&msDouble, start, stop));

    // Copy results back to host (optional verification)
    CUDA_CHECK(cudaMemcpy(h_c_f, d_c_f, sizeFloat, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_d, d_c_d, sizeDouble, cudaMemcpyDeviceToHost));

    // Print timings
    printf("Float kernel time:  %.3f ms\n", msFloat);
    printf("Double kernel time: %.3f ms\n", msDouble);
    printf("Speed ratio (float/double): %.2f\n", msDouble / msFloat);

    // Cleanup
    free(h_a_f); free(h_b_f); free(h_c_f);
    free(h_a_d); free(h_b_d); free(h_c_d);

    CUDA_CHECK(cudaFree(d_a_f));
    CUDA_CHECK(cudaFree(d_b_f));
    CUDA_CHECK(cudaFree(d_c_f));
    CUDA_CHECK(cudaFree(d_a_d));
    CUDA_CHECK(cudaFree(d_b_d));
    CUDA_CHECK(cudaFree(d_c_d));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
