```cuda
/*
Aim of the program: Implement vector addition for double precision floating-point numbers instead of float's.

Thinking process:
1. Understand the requirement: replace float-based vector addition with double precision.
2. CUDA supports double type on devices with compute capability >= 1.3, which is common.
3. The kernel will take two input pointers (double*) and one output pointer (double*), plus the number of elements N.
4. Use a simple grid-stride loop to handle arbitrary N efficiently.
5. Host code will:
   - Allocate host arrays of type double and fill them with test values.
   - Allocate device memory.
   - Copy host data to device.
   - Launch kernel with appropriate block/grid dimensions.
   - Copy result back to host.
   - Verify correctness by comparing with CPU reference.
   - Free memory and exit.
6. Error handling: wrap CUDA calls with a macro to check for errors.
7. Use `nvcc` compile flags: `nvcc -arch=sm_52 vector_add_double.cu -o vector_add_double`.
8. The code will be self-contained and ready to compile/run on a compatible GPU.
9. Use constants for block size and compute grid size based on N.
10. The kernel should be straightforward: each thread processes one element (or more via stride loop).
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    }

__global__ void vectorAddDouble(const double *a, const double *b, double *c, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride)
    {
        c[i] = a[i] + b[i];
    }
}

int main(void)
{
    const size_t N = 1 << 20; // Example size: 1M elements
    const size_t size = N * sizeof(double);

    // Allocate host memory
    double *h_a = (double *)malloc(size);
    double *h_b = (double *)malloc(size);
    double *h_c = (double *)malloc(size);
    if (!h_a || !h_b || !h_c)
    {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<double>(i);
        h_b[i] = static_cast<double>(N - i);
    }

    // Allocate device memory
    double *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void **)&d_a, size));
    CHECK_CUDA(cudaMalloc((void **)&d_b, size));
    CHECK_CUDA(cudaMalloc((void **)&d_c, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddDouble<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify result
    bool success = true;
    for (size_t i = 0; i < N; ++i)
    {
        double expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-9)
        {
            fprintf(stderr, "Verification failed at index %zu: GPU %f, CPU %f\n",
                    i, h_c[i], expected);
            success = false;
            break;
        }
    }

    if (success)
    {
        printf("Vector addition successful. All results verified.\n");
    }
    else
    {
        printf("Vector addition failed.\n");
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```