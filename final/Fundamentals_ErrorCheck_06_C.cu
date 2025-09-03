```cpp
/*
Create a macro `CHECK_KERNEL(kernel_launch)` that expands to `{ kernel_launch; checkKernelLaunch(...); }`.

I considered how to design a reusable CUDA error-checking macro that can be used with any kernel launch statement. The macro should enclose the launch in braces so that it can be used as a single statement in control structures. After the launch, we need to invoke a helper function that checks for errors from both the launch itself (via cudaGetLastError) and any subsequent execution failure (via cudaDeviceSynchronize). The helper function `checkKernelLaunch` is defined to perform these checks and report any problems. The macro is defined as `CHECK_KERNEL(kernel_launch)` and expands to `{ kernel_launch; checkKernelLaunch(); }` exactly as requested. The code below provides the macro definition, the helper function, and a simple test kernel to demonstrate its usage.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_KERNEL(kernel_launch) { kernel_launch; checkKernelLaunch(); }

// Simple kernel for demonstration
__global__ void addKernel(const int *a, const int *b, int *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

// Helper function to check for CUDA errors after kernel launch
void checkKernelLaunch()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Wait for kernel to finish and check for runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA runtime error after kernel execution: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(int);

    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Use the macro to launch the kernel
    CHECK_KERNEL(addKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N));

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify result
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != N) { ok = false; break; }
    }
    printf("Result %s\n", ok ? "OK" : "FAIL");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```