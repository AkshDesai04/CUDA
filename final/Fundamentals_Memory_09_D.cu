/*
What is the downside of using pinned memory? (It's a limited resource and reduces memory available to the OS and other applications).

Thinking:
The prompt asks for a CUDA program that demonstrates the use of pinned (page-locked) host memory, and explicitly mentions the downside of using pinned memory: it is a limited resource and reduces memory available to the OS and other applications.  To satisfy the requirement we will:

1. Provide a complete .cu file that can compile with nvcc.
2. Include a multiline comment at the very top that contains the exact phrase requested in the prompt and also explains the reasoning behind the code – this comment will serve as the "aim of the program" and will include the entire thinking process.
3. Write a simple vector‑addition kernel so we can actually transfer data to/from the GPU.
4. Allocate host memory using `cudaHostAlloc` (pinned memory), copy data to device, launch the kernel, copy results back, and free the resources.
5. After the kernel execution we will print a message explaining that the pinned memory usage reduces the amount of memory available to the OS and other processes, and that it is a limited resource.
6. All error checking will be done via a helper macro so the code stays readable.
7. No other text or files will be output – only the .cu file contents.

The program is intentionally simple so that it can be understood quickly while still illustrating the downside of pinned memory.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Simple vector addition kernel
__global__ void vecAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main(void)
{
    const int N = 1 << 20;          // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate pinned (page-locked) host memory for input and output
    float *h_a, *h_b, *h_c;
    CUDA_CHECK(cudaHostAlloc((void**)&h_a, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_b, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_c, bytes, cudaHostAllocDefault));

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // Transfer data from host (pinned) to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // Transfer results back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Simple verification
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
            ok = false;
            break;
        }
    }
    printf("Verification %s\n", ok ? "PASSED" : "FAILED");

    // Free resources
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));

    // Downside reminder
    printf("\nNote: The use of pinned (page‑locked) host memory above consumes\n"
           "a contiguous block of physical memory that the OS cannot repurpose.\n"
           "Pinned memory is a limited resource; allocating large amounts can\n"
           "reduce the memory available to the OS and other applications.\n");

    return 0;
}
