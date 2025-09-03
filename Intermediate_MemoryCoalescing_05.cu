```cuda
/* 
   Aim of the program: 
   Write a kernel that accesses a struct array: `struct { float x, y, z; }`. 
   Compare the performance of an Array of Structs (AoS) access pattern (`my_structs[i].x`) 
   versus a Struct of Arrays (SoA) pattern (`my_xs[i]`, `my_ys[i]`, `my_zs[i]`). 
   SoA is usually much better for coalescing.

   Thinking:
   1. Define a simple 3-component vector struct `Vec3` with `float x, y, z;`.
   2. Allocate an array of `Vec3` (AoS) and parallel arrays of floats (SoA) on the GPU.
   3. Write two kernels: one that reads from the AoS array and one that reads from the SoA arrays.
      Each kernel will perform a simple arithmetic operation to ensure memory access.
   4. Measure execution time for each kernel using CUDA events to highlight coalescing differences.
   5. Keep the problem size large enough (e.g., 1<<20 elements) to observe timing differences.
   6. Use basic error checking macro for CUDA API calls.
   7. Provide a `main` that initializes data, runs both kernels, times them, and prints the results.
   8. Ensure the code is self-contained and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Vector struct for AoS */
struct Vec3 {
    float x;
    float y;
    float z;
};

/* AoS kernel: reads from struct array */
__global__ void kernel_aos(const Vec3* __restrict__ a, float* __restrict__ out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const Vec3 v = a[idx];
        out[idx] = v.x * v.y + v.z;  // simple operation
    }
}

/* SoA kernel: reads from separate float arrays */
__global__ void kernel_soa(const float* __restrict__ xs,
                           const float* __restrict__ ys,
                           const float* __restrict__ zs,
                           float* __restrict__ out,
                           size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = xs[idx];
        float y = ys[idx];
        float z = zs[idx];
        out[idx] = x * y + z;  // same operation
    }
}

int main(void)
{
    const size_t N = 1 << 20;          // number of elements (~1M)
    const size_t SIZE = N * sizeof(float);
    const size_t AO_SIZE = N * sizeof(Vec3);

    /* Host memory allocation */
    Vec3* h_aos = (Vec3*)malloc(AO_SIZE);
    float* h_xs = (float*)malloc(SIZE);
    float* h_ys = (float*)malloc(SIZE);
    float* h_zs = (float*)malloc(SIZE);
    float* h_out = (float*)malloc(SIZE);

    /* Initialize data */
    for (size_t i = 0; i < N; ++i) {
        h_aos[i].x = (float)i * 0.001f;
        h_aos[i].y = (float)(i + 1) * 0.002f;
        h_aos[i].z = (float)(i + 2) * 0.003f;

        h_xs[i] = h_aos[i].x;
        h_ys[i] = h_aos[i].y;
        h_zs[i] = h_aos[i].z;
    }

    /* Device memory allocation */
    Vec3* d_aos;
    float* d_xs;
    float* d_ys;
    float* d_zs;
    float* d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_aos, AO_SIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_xs, SIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_ys, SIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_zs, SIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_out, SIZE));

    /* Copy data to device */
    CUDA_CHECK(cudaMemcpy(d_aos, h_aos, AO_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs, h_xs, SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, h_ys, SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_zs, h_zs, SIZE, cudaMemcpyHostToDevice));

    /* Kernel launch configuration */
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Events for timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Run AoS kernel */
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernel_aos<<<blocks, threadsPerBlock>>>(d_aos, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_aos = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time_aos, start, stop));

    /* Run SoA kernel */
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernel_soa<<<blocks, threadsPerBlock>>>(d_xs, d_ys, d_zs, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_soa = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time_soa, start, stop));

    /* Copy result back (not used but ensures kernel finished) */
    CUDA_CHECK(cudaMemcpy(h_out, d_out, SIZE, cudaMemcpyDeviceToHost));

    /* Print timings */
    printf("AoS kernel time:  %f ms\n", time_aos);
    printf("SoA kernel time:  %f ms\n", time_soa);

    /* Clean up */
    free(h_aos);
    free(h_xs);
    free(h_ys);
    free(h_zs);
    free(h_out);
    CUDA_CHECK(cudaFree(d_aos));
    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_zs));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```