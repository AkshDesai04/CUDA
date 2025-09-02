/*
Implement DAXPY using `double`s.

Thought process:
- DAXPY is the vector operation Y := a*X + Y, where a is scalar, X and Y are double-precision vectors.
- We want a CUDA implementation that performs this operation on the GPU.
- The kernel will compute each element independently, so we can launch one thread per element.
- Need to handle boundary: if the total number of threads exceeds array size, guard with `if (i < N)`.
- Host side: allocate arrays X and Y, initialize them, allocate device memory, copy data, launch kernel, copy back results.
- Add simple CPU verification to ensure correctness.
- Use simple timing to show GPU performance.
- Use error checking macros for CUDA API calls.
- Include main function with optional command-line arguments: size N and scalar a, otherwise defaults.
- Provide minimal output: time measurements, first few elements for sanity check.
- Ensure double precision by using double type throughout.
*/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

__global__ void daxpy_kernel(double a, const double *X, double *Y, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        Y[idx] = a * X[idx] + Y[idx];
    }
}

int main(int argc, char *argv[])
{
    // Default parameters
    int N = 1 << 20;          // 1M elements
    double a = 2.5;

    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) a = atof(argv[2]);

    // Host memory allocation
    double *h_X = (double*)malloc(N * sizeof(double));
    double *h_Y = (double*)malloc(N * sizeof(double));
    if (!h_X || !h_Y)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    srand((unsigned)time(nullptr));
    for (int i = 0; i < N; ++i)
    {
        h_X[i] = static_cast<double>(rand()) / RAND_MAX;
        h_Y[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // Device memory allocation
    double *d_X = nullptr;
    double *d_Y = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, N * sizeof(double), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Timing GPU execution
    auto gpu_start = std::chrono::high_resolution_clock::now();

    daxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, d_X, d_Y, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;
    printf("GPU DAXPY time: %.3f ms\n", gpu_time.count());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, N * sizeof(double), cudaMemcpyDeviceToHost));

    // CPU reference implementation for verification
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
    {
        h_X[i] = a * h_X[i] + h_Y[i]; // Note: we modify h_X to hold the result for comparison
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    printf("CPU DAXPY time: %.3f ms\n", cpu_time.count());

    // Verify results
    bool ok = true;
    for (int i = 0; i < N; ++i)
    {
        double diff = fabs(h_X[i] - h_Y[i]);
        if (diff > 1e-9)
        {
            ok = false;
            printf("Mismatch at index %d: CPU=%f, GPU=%f, diff=%e\n", i, h_X[i], h_Y[i], diff);
            break;
        }
    }
    printf("Verification: %s\n", ok ? "PASSED" : "FAILED");

    // Clean up
    free(h_X);
    free(h_Y);
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));

    return 0;
}
