```cuda
/* 
Aim of the program: 
Modify the function to also call `cudaGetLastError()` first to catch any launch configuration errors.

Thinking:
- The typical CUDA pattern is to launch a kernel and then immediately call cudaGetLastError() to detect errors
  that may have occurred during kernel launch (e.g., bad block/warp configuration).
- The user explicitly requests that `cudaGetLastError()` be called *before* the kernel launch as well.
- Calling `cudaGetLastError()` before a launch is useful for clearing any lingering error state from
  previous CUDA API calls and ensuring the subsequent launch is not affected by stale error codes.
- We will implement a helper function `launchKernel()` that first calls `cudaGetLastError()` to clear
  any previous errors, then launches a simple kernel (`addKernel`) and calls `cudaGetLastError()` again
  to detect any launch configuration errors.
- The kernel will perform a trivial element‑wise addition of two arrays.
- We will add proper error checking after memory allocation, kernel launch, and memory copy.
- The main function will set up test data, invoke `launchKernel()`, and verify the results.

The following code demonstrates these concepts in a self‑contained CUDA program.
*/
#include <stdio.h>
#include <cuda_runtime.h>

// Simple element‑wise addition kernel
__global__ void addKernel(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

// Wrapper that calls cudaGetLastError() before and after the kernel launch
void launchKernel(const float* d_a, const float* d_b, float* d_c, int n)
{
    // Clear any previous errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error before kernel launch: %s\n", cudaGetErrorString(err));
        return;
    }

    // Define launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Check for launch configuration errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Optionally, synchronize to catch runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    // Initialize host data
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = i * 1.0f;
        h_b[i] = (N - i) * 1.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaError_t err = cudaMalloc((void**)&d_a, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_a failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc((void**)&d_b, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_b failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc((void**)&d_c, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_c failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Copy inputs to device
    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_a failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_b failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Launch kernel via wrapper
    launchKernel(d_a, d_b, d_c, N);

    // Copy result back to host
    err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_c to h_c failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Verify a few results
    bool ok = true;
    for (int i = 0; i < 10; ++i)
    {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f)
        {
            fprintf(stderr, "Mismatch at %d: expected %f, got %f\n", i, expected, h_c[i]);
            ok = false;
            break;
        }
    }
    if (ok)
        printf("First 10 results are correct.\n");
    else
        printf("Verification failed.\n");

    // Free resources
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```