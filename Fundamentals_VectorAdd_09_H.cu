/*
Pass all scalars and vector pointers to the kernel.
In this CUDA program we demonstrate how to pass both scalar arguments (such as array size and a scaling factor)
and vector pointers (device memory addresses) to a kernel. The kernel performs a simple element-wise operation
on two input arrays and stores the result in a third array, applying a scalar multiplier. All relevant arguments
are explicitly passed into the kernel launch configuration. The main host code allocates and initializes
device memory, launches the kernel with appropriate grid/block dimensions, retrieves the computed results,
and finally cleans up the allocated resources. This example illustrates the mechanics of argument passing
in CUDA and serves as a template for more complex kernels that require multiple scalar and pointer inputs.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that adds two vectors element-wise with a scalar multiplier
__global__ void add_kernel(const float* a, const float* b, float* c, int n, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = alpha * (a[idx] + b[idx]);
    }
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                    \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

int main(void)
{
    const int n = 1 << 20; // 1M elements
    const size_t size = n * sizeof(float);
    const float alpha = 2.5f;

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < n; ++i)
    {
        h_a[i] = (float)i;
        h_b[i] = (float)(n - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel, passing all scalars and vector pointers
    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n, alpha);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("Sample results:\n");
    for (int i = 0; i < 5; ++i)
    {
        printf("c[%d] = %f\n", i, h_c[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
