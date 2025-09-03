```cuda
/* 
Instead of an if, can you use arithmetic to prevent out-of-bounds access (e.g. C[index] = ... where index is somehow clamped)? Is this better or worse than a branch?

Thinking:
- A branch such as `if (idx < N) C[idx] = a[idx] + b[idx];` protects against out-of-bounds writes but can cause divergence if some threads in a warp hit the branch and others do not. In modern GPUs, divergence is mitigated but still incurs extra serial execution of both paths.
- An arithmetic clamping approach rewrites the index so that it always stays within bounds: `int clamped = min(idx, N-1); C[clamped] = a[clamped] + b[clamped];`. This eliminates branching but introduces a conditional (min) that may be compiled to a branch internally. Compilers can sometimes replace `min` with a branchless max operation using `__clamp`. However, the overhead of extra instructions may outweigh the branch penalty if the branch is rarely taken.
- The best choice depends on the workload: if many threads hit the out-of-bounds case, clamping is beneficial; if only a few threads do, a branch may be cheaper. Benchmarking is required for definitive answer.
- In this example we will implement both versions and run them on a small vector to illustrate the difference. 
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_with_branch(float* a, float* b, float* c, int N, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += stride; // deliberately create out-of-bounds access for some threads
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_with_clamp(float* a, float* b, float* c, int N, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += stride;
    // Clamp idx to [0, N-1] using a branchless approach
    int clamped = idx < N ? idx : N - 1; // could be replaced by __clamp for newer GPUs
    c[clamped] = a[clamped] + b[clamped];
}

int main()
{
    const int N = 1024;
    const int stride = 100; // makes many threads out-of-bounds

    float *h_a, *h_b, *h_c_branch, *h_c_clamp;
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c_branch = (float*)malloc(N * sizeof(float));
    h_c_clamp = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i)
    {
        h_a[i] = i * 1.0f;
        h_b[i] = (N - i) * 1.0f;
    }

    float *d_a, *d_b, *d_c_branch, *d_c_clamp;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c_branch, N * sizeof(float));
    cudaMalloc(&d_c_clamp, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Run branch version
    add_with_branch<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c_branch, N, stride);
    cudaMemcpy(h_c_branch, d_c_branch, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Run clamp version
    add_with_clamp<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c_clamp, N, stride);
    cudaMemcpy(h_c_clamp, d_c_clamp, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print first 10 results for each
    printf("First 10 results (branch):\n");
    for (int i = 0; i < 10; ++i)
        printf("%f ", h_c_branch[i]);
    printf("\n");

    printf("First 10 results (clamp):\n");
    for (int i = 0; i < 10; ++i)
        printf("%f ", h_c_clamp[i]);
    printf("\n");

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_branch);
    cudaFree(d_c_clamp);
    free(h_a);
    free(h_b);
    free(h_c_branch);
    free(h_c_clamp);

    return 0;
}
```